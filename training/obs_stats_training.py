"""Observation statistics collected during real PPO training.

Trains iter/hybrid/flattop like ``train_ppo.py``, with an ``ObsStatsWrapper``
that tracks running per-sensor mean/variance (Welford) of every observation
the agent receives - measuring natural variation under a real, learning
policy, unlike the zero/random-action proxies in
``experiments/studies/signal_to_noise_ratio``.

Buckets one per eval checkpoint (ceil(total_timesteps / eval_freq)), so
train bucket i and eval checkpoint i cover the same training window.
Training and eval stay separate automatically: rejax's evaluator builds
and discards its own env states via ``env.reset``, never touching the
training accumulator.

Statistics use the *degraded* observation space (what the policy actually
sees), not the full-resolution space NoiseWrapper operates on. Run with
``--env.noise-multiplier 0.0`` to measure natural variation uncontaminated
by injected noise.

Logs through the same ``SeedBufferLogger`` as ``train_ppo.py``, so progress
and ``evaluation/*``/``train/*`` metrics look identical, with per-sensor
tables printed live at each checkpoint (seed-averaged) and a final TOTAL
table over the whole run.

Run:
    uv run python training/obs_stats_training.py \\
        --env.noise-multiplier 0.0 \\
        --ppo.total-timesteps 5120000 --ppo.eval-freq 512000 \\
        --num-seeds 3
"""

from __future__ import annotations

import dataclasses
import math
import operator
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from envelope import WrappedState, Wrapper, field, static_field

from experiments.plotting.wandb_logging import (
    _base_metrics,
    _collect_returns_and_lengths,
)
from experiments.studies.baseline_study import seed_keys
from plasmax.environment.factory import make
from plasmax.wrappers import (
    NoiseWrapper,
    ObsDelayWrapper,
    ObsFilterWrapper,
    PhysicsRandomizationWrapper,
    QuantizeActionWrapper,
    SensorNoiseConfig,
    TruncationWrapper,
    _training_wrappers,
)
from training.train_ppo import Config, _build_algo, _run_name
from training.vmap_logging import SeedBufferLogger

# ---------------------------------------------------------------------------
# Statistics wrapper
# ---------------------------------------------------------------------------


class ObsStatsState(WrappedState):
    """Inner state plus per-bucket Welford accumulators and a step counter."""

    count: jax.Array = field()  # (n_buckets,)
    mean: jax.Array = field()  # (n_buckets, obs_dim)
    m2: jax.Array = field()  # (n_buckets, obs_dim)
    step: jax.Array = field()  # scalar


class ObsStatsWrapper(Wrapper):
    """Accumulate per-sensor mean/variance of emitted observations.

    Each vmapped environment instance keeps its own accumulator, bucketed by
    that instance's own step count so bucket ``i`` covers exactly the same
    training window as eval checkpoint ``i`` - the same span of steps that
    fed into the policy being evaluated at that checkpoint. Accumulators
    are carried through ``reset`` - an episode ending must not discard the
    statistics gathered so far.

    ``m2`` is the sum of squared deviations from the running mean (Welford),
    which is numerically stable over millions of updates and can be merged
    across instances exactly. Variance is ``m2 / count``.
    """

    bucket_size: int = static_field(default=1)
    n_buckets: int = static_field(default=1)

    @property
    def _obs_dim(self) -> int:
        return int(self.env.observation_space.shape[0])

    def _fresh(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        zeros = jnp.zeros((self.n_buckets, self._obs_dim), dtype=jnp.float32)
        return (
            jnp.zeros((self.n_buckets,), dtype=jnp.float32),
            zeros,
            zeros,
            jnp.zeros((), dtype=jnp.int32),
        )

    def init(self, key):
        inner_state, info = self.env.init(key)
        count, mean, m2, step = self._fresh()
        return (
            ObsStatsState(
                inner_state=inner_state, count=count, mean=mean, m2=m2, step=step
            ),
            info,
        )

    def reset(self, state: ObsStatsState, key):
        # Carry the accumulator across the episode boundary; only the
        # wrapped environment's own state is rebuilt.
        inner_state, info = self.env.reset(state.inner_state, key)
        return (
            ObsStatsState(
                inner_state=inner_state,
                count=state.count,
                mean=state.mean,
                m2=state.m2,
                step=state.step,
            ),
            info,
        )

    def step(self, state: ObsStatsState, action):
        inner_state, info = self.env.step(state.inner_state, action)
        # A solver-failure disruption (core.py's termination_code=3) can
        # return a non-finite obs; core.py guards reward against this same
        # case (`reward = jnp.where(disruption, ...)`), but not obs itself.
        # A finite disruption boundary (q_min/Greenwald) is real, policy-
        # relevant data and stays in; only non-finite steps are skipped.
        valid = jnp.all(jnp.isfinite(info.obs))
        bucket = jnp.minimum(state.step // self.bucket_size, self.n_buckets - 1)

        count_b = state.count[bucket] + 1.0
        delta = info.obs - state.mean[bucket]
        mean_b = state.mean[bucket] + delta / count_b
        m2_b = state.m2[bucket] + delta * (info.obs - mean_b)

        next_state = ObsStatsState(
            inner_state=inner_state,
            count=state.count.at[bucket].set(
                jnp.where(valid, count_b, state.count[bucket])
            ),
            mean=state.mean.at[bucket].set(
                jnp.where(valid, mean_b, state.mean[bucket])
            ),
            m2=state.m2.at[bucket].set(jnp.where(valid, m2_b, state.m2[bucket])),
            step=state.step + 1,
        )
        return next_state, info


def _find_stats_state(tree) -> ObsStatsState:
    """Locate the ObsStatsState nested somewhere inside a training state."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ObsStatsState):
            return node
        for attr in ("inner_state", "env_state", "state"):
            child = getattr(node, attr, None)
            if child is not None:
                stack.append(child)
    raise RuntimeError("ObsStatsState not found in training state")


# ---------------------------------------------------------------------------
# Merging accumulators
# ---------------------------------------------------------------------------


def _merge(count, mean, m2, axis):
    """Chan's parallel merge of Welford accumulators along ``axis``.

    Shapes are ``(..., N, ...)`` with ``N`` the axis being reduced. Buckets
    with zero samples contribute nothing and are guarded against division by
    zero.
    """
    total = np.sum(count, axis=axis)
    safe = np.where(total == 0.0, 1.0, total)
    merged_mean = np.sum(count * mean, axis=axis) / safe
    spread = (mean - np.expand_dims(merged_mean, axis)) ** 2
    merged_m2 = np.sum(m2 + count * spread, axis=axis)
    return total, merged_mean, merged_m2


def _std(count, m2):
    safe = np.where(count == 0.0, 1.0, count)
    return np.sqrt(np.maximum(m2 / safe, 0.0))


def _merge_jax(count, mean, m2, axis):
    """In-trace equivalent of :func:`_merge`, for use inside the callback."""
    total = jnp.sum(count, axis=axis)
    safe = jnp.where(total == 0.0, 1.0, total)
    merged_mean = jnp.sum(count * mean, axis=axis) / safe
    spread = (mean - jnp.expand_dims(merged_mean, axis)) ** 2
    merged_m2 = jnp.sum(m2 + count * spread, axis=axis)
    return total, merged_mean, merged_m2


def _obs_metrics(prefix, layout, names, noise_cfg, count, mean, m2):
    """Flat ``{metric_name: scalar}`` dict for one bucket's statistics.

    ``count``/``mean``/``m2`` are per-observation-channel arrays of shape
    ``(obs_dim,)``. Channels belonging to one sensor are merged so each
    sensor contributes a single mean/std, matching the end-of-run tables.
    """
    metrics = {}
    for name in names:
        sl = layout.slice_of(name)
        total, _, sensor_m2 = _merge_jax(count[sl], mean[sl], m2[sl], axis=0)
        safe = jnp.where(total == 0.0, 1.0, total)
        obs_std = jnp.sqrt(jnp.maximum(sensor_m2 / safe, 0.0))
        abs_mean = jnp.mean(jnp.abs(mean[sl]))
        noise_std = float(noise_cfg.get(name, 0.0)) * abs_mean
        metrics[f"{prefix}/{name}/obs_abs_mean"] = abs_mean
        metrics[f"{prefix}/{name}/obs_std"] = obs_std
        metrics[f"{prefix}/{name}/noise_std_at_1x"] = noise_std
        metrics[f"{prefix}/{name}/noise_to_signal"] = jnp.where(
            obs_std > 0.0, noise_std / jnp.where(obs_std > 0.0, obs_std, 1.0), jnp.nan
        )
    return metrics


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def _build_env(cfg: Config, bucket_size: int, n_buckets: int):
    """RealisticWrappers, plus noise scaling and the statistics wrapper.

    Mirrors ``plasmax.wrappers.RealisticWrappers`` exactly, with two
    additions: sensor noise magnitudes are scaled by
    ``--env.noise-multiplier``, and ObsStatsWrapper is inserted after the
    observation degradations (so it sees the degraded observation) but
    before the training wrappers (so it is unaffected by action rescaling).
    """
    env = make(
        cfg.env.env_setup,
        cfg.env.backend,
        reward=cfg.env.reward,
        disruption_penalty=cfg.env.disruption_penalty,
    )
    plasmax_cfg = env.plasmax_config
    if plasmax_cfg.physics_randomization:
        env = PhysicsRandomizationWrapper(env)

    real = plasmax_cfg.observations.realistic
    if real.noise:
        scaled = {
            name: value * cfg.env.noise_multiplier for name, value in real.noise.items()
        }
        layout = env.obs_layout()
        sensors = layout.profile_names + layout.scalar_names
        noise_scale = SensorNoiseConfig(
            relative_std={k: v for k, v in scaled.items() if k in sensors}
        ).to_noise_scale(layout)
        env = NoiseWrapper(env, noise_scale=noise_scale)
    if real.resolution:
        env = ObsFilterWrapper.from_resolution_config(env)
    if real.filter is not None:
        env = ObsFilterWrapper(env)
    if real.delay:
        env = ObsDelayWrapper(env)

    env = ObsStatsWrapper(env, bucket_size=bucket_size, n_buckets=n_buckets)

    env = _training_wrappers(env, cfg.env.time_aware)
    if cfg.env.quantize_bins is not None:
        bins = operator.index(cfg.env.quantize_bins)
        env = QuantizeActionWrapper(env, (bins,) * len(env.actuator_specs))
    elif plasmax_cfg.actions.realistic.quantize:
        env = QuantizeActionWrapper(env)
    return TruncationWrapper(env)


def _sensor_slices(env):
    """Sensor name -> slice in the degraded observation vector."""
    inner = env
    while not isinstance(inner, ObsStatsWrapper):
        inner = inner.env
    layout = inner.env.obs_layout()
    names = layout.profile_names + layout.scalar_names
    return layout, names


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------


def _make_logging_callback(
    logger,
    *,
    layout,
    names,
    noise_cfg,
    bucket_size,
    n_buckets,
    num_steps,
    n_seeds,
    eval_rng,
    deterministic,
):
    """``run_idx -> callback`` factory, mirroring make_buffered_seed_callback.

    Emits exactly the metrics a normal ``train_ppo.py`` run emits, plus the
    observation statistics for the training bucket that just finished
    (``obs_train/*``, read out of ``ts.env_state``) and for this checkpoint's
    own eval rollouts (``obs_eval/*``, computed from the eval trajectory).
    Routing through ``logger.log`` via ``jax.debug.callback`` means it
    appears live during training rather than only at the end. The tables
    themselves are printed by :class:`_TableLogger` at flush time, so they
    are seed-averaged and stay ordered with the ``step=`` progress line.
    """

    def for_run(run_idx):
        def _callback(algo, ts, rng, train_metrics):
            traj, episode_returns, episode_lengths = _collect_returns_and_lengths(
                algo,
                ts,
                eval_rng if eval_rng is not None else rng,
                num_steps,
                n_seeds,
                lean=True,
                deterministic=deterministic,
            )
            metrics = _base_metrics(
                traj, episode_returns, episode_lengths, train_metrics
            )

            # --- training bucket that just completed -------------------
            stats = _find_stats_state(ts.env_state)
            # (num_envs, n_buckets[, obs_dim]) -> merge over envs.
            count = stats.count[..., None] * jnp.ones_like(stats.mean)
            train_count, train_mean, train_m2 = _merge_jax(
                count, stats.mean, stats.m2, axis=0
            )
            # Buckets fill in order, so the one just closed is step//size - 1.
            # Clipped for the pre-training checkpoint, where none are filled
            # yet and the reported counts are simply zero.
            idx = jnp.clip(stats.step[0] // bucket_size - 1, 0, n_buckets - 1)
            metrics.update(
                _obs_metrics(
                    "obs_train",
                    layout,
                    names,
                    noise_cfg,
                    train_count[idx],
                    train_mean[idx],
                    train_m2[idx],
                )
            )

            # --- this checkpoint's eval rollouts -----------------------
            valid = traj.valid[..., None]
            eval_count = jnp.maximum(jnp.sum(traj.valid).astype(jnp.float32), 1.0)
            eval_mean = jnp.sum(jnp.where(valid, traj.obs, 0.0), axis=(0, 1)) / eval_count
            eval_m2 = jnp.sum(
                jnp.where(valid, (traj.obs - eval_mean) ** 2, 0.0), axis=(0, 1)
            )
            eval_count_vec = jnp.full_like(eval_mean, eval_count)
            metrics.update(
                _obs_metrics(
                    "obs_eval",
                    layout,
                    names,
                    noise_cfg,
                    eval_count_vec,
                    eval_mean,
                    eval_m2,
                )
            )
            jax.debug.callback(logger.log, ts.global_step, run_idx, metrics)
            return episode_returns, episode_lengths

        return _callback

    return for_run


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _sensor_rows(layout, names, noise_cfg, count, mean, m2):
    """Per-sensor aggregate over each sensor's slice of the obs vector."""
    rows = []
    for name in names:
        sl = layout.slice_of(name)
        # Pool the sensor's channels: identical counts, so a plain merge
        # across the slice gives the sensor-level mean/std.
        c_slice = np.broadcast_to(count, mean.shape)[sl]
        total, _, sensor_m2 = _merge(c_slice, mean[sl], m2[sl], axis=0)
        sensor_std = float(_std(total, sensor_m2))
        abs_mean = float(np.mean(np.abs(mean[sl])))
        rel_std = float(noise_cfg.get(name, 0.0))
        noise_std = rel_std * abs_mean
        ratio = noise_std / sensor_std if sensor_std > 0 else float("nan")
        rows.append(
            {
                "sensor": name,
                "noise_rel_std": rel_std,
                "obs_abs_mean": abs_mean,
                "obs_std": sensor_std,
                "noise_std_at_1x": noise_std,
                "noise_to_signal": ratio,
            }
        )
    return rows


_HEADERS = (
    ("sensor", "<18", "s"),
    ("noise_rel_std", ">14", ".3f"),
    ("obs_abs_mean", ">14", ".4g"),
    ("obs_std", ">12", ".4g"),
    ("noise_std_at_1x", ">16", ".4g"),
    ("noise_to_signal", ">16", ".3f"),
)


def _print_table(title: str, rows) -> None:
    print(f"\n{title}")
    print("".join(f"{name:{width}}" for name, width, _ in _HEADERS))
    for row in rows:
        print(
            "".join(
                f"{row[name]:{width}{fmt}}" if fmt != "s" else f"{row[name]:{width}}"
                for name, width, fmt in _HEADERS
            )
        )


class _TableLogger(SeedBufferLogger):
    """SeedBufferLogger that also prints per-sensor tables at flush time.

    The observation statistics already travel in the metrics dict, so the
    tables are rebuilt from the seed-averaged values rather than printed
    per-seed. Printing here (rather than from a separate debug callback)
    keeps each table adjacent to its own ``step=`` progress line instead of
    racing it - ``jax.debug.callback`` ordering across a vmapped seed axis
    is not guaranteed.
    """

    def __init__(self, *args, layout, names, noise_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self._layout = layout
        self._names = names
        self._noise_cfg = noise_cfg

    def _rows_from_metrics(self, prefix, mean):
        rows = []
        for name in self._names:
            key = f"{prefix}/{name}"
            if f"{key}/obs_std" not in mean:
                return None
            rows.append(
                {
                    "sensor": name,
                    "noise_rel_std": float(self._noise_cfg.get(name, 0.0)),
                    "obs_abs_mean": mean[f"{key}/obs_abs_mean"],
                    "obs_std": mean[f"{key}/obs_std"],
                    "noise_std_at_1x": mean[f"{key}/noise_std_at_1x"],
                    "noise_to_signal": mean[f"{key}/noise_to_signal"],
                }
            )
        return rows

    def _flush_step(self, step: int) -> None:
        # Read the buffer before super() pops it, so the tables print above
        # the step= line that super() emits.
        per_seed = self._buffers.get(step, {})
        if per_seed:
            keys = list(next(iter(per_seed.values())).keys())
            mean = {
                k: float(np.mean([per_seed[r][k] for r in sorted(per_seed)]))
                for k in keys
            }
            elapsed = time.time() - (self.start_time or time.time())
            # At step 0 no training bucket has closed, so only eval is real.
            if step > 0:
                rows = self._rows_from_metrics("obs_train", mean)
                if rows:
                    _print_table(f"TRAIN bucket @ step {step:,}", rows)
            rows = self._rows_from_metrics("obs_eval", mean)
            if rows:
                label = "pre-training" if step == 0 else f"step {step:,}"
                _print_table(f"EVAL checkpoint @ {label} (t+{elapsed:.0f}s)", rows)
            print()
        super()._flush_step(step)


def main(cfg: Config) -> None:
    if cfg.env.noise_multiplier != 0.0:
        print(
            f"warning: noise_multiplier={cfg.env.noise_multiplier}; obs_std will "
            "include injected noise. Use 0.0 to measure natural variation."
        )

    # Matches rejax's own checkpoint count: ceil(total_timesteps / eval_freq).
    n_buckets = max(1, math.ceil(cfg.ppo.total_timesteps / cfg.ppo.eval_freq))
    # Per-env steps between checkpoints - eval_freq is a total across envs.
    bucket_size = max(1, cfg.ppo.eval_freq // cfg.ppo.num_envs)

    env = _build_env(cfg, bucket_size, n_buckets)
    layout, names = _sensor_slices(env)
    noise_cfg = env.plasmax_config.observations.realistic.noise

    run_name = f"{_run_name(cfg)}-obsstats"
    logger = _TableLogger(
        num_seeds=cfg.num_seeds,
        run_name=run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=dataclasses.asdict(cfg),
        out_dir=cfg.history_dir,
        seed_ids=tuple(range(cfg.seed, cfg.seed + cfg.num_seeds)),
        job_type=cfg.algorithm,
        tags=cfg.wandb.tags,
        layout=layout,
        names=names,
        noise_cfg=noise_cfg,
    )

    # Reuse train_ppo's builder so agent_kwargs/env_params/normalisation
    # options stay in sync with the launcher rather than drifting.
    algo = _build_algo(cfg, env)
    for_run = _make_logging_callback(
        logger,
        layout=layout,
        names=names,
        noise_cfg=noise_cfg,
        bucket_size=bucket_size,
        n_buckets=n_buckets,
        num_steps=algo.env_params.max_steps_in_episode,
        n_seeds=cfg.env.eval_n_envs,
        eval_rng=jax.random.PRNGKey(cfg.env.eval_seed),
        deterministic=cfg.env.deterministic_eval,
    )

    # rejax evaluates once before training, but from outside its lax.scan -
    # with no data dependency on the scan, XLA is free to schedule that
    # callback after later ones, so it lands out of order in the log. Run it
    # explicitly here instead, and tell rejax to skip its own.
    algo = algo.replace(skip_initial_evaluation=True)

    def pre_eval(rng, run_idx):
        algo_i = algo.with_eval_callback(for_run(run_idx))
        ts0 = algo_i.init_state(rng)
        return algo_i.eval_callback(algo_i, ts0, ts0.rng)

    def train_one(rng, run_idx):
        # Rebuilt inside the trace so each vmapped seed's run_idx tracer
        # reaches logger.log, exactly as train_ppo's _run_vmap does.
        return algo.with_eval_callback(for_run(run_idx)).train(rng)

    seeds = seed_keys(cfg.seed, cfg.num_seeds)
    run_idxs = jnp.arange(cfg.num_seeds)
    train_fn = jax.jit(jax.vmap(train_one))

    print(f"Lowering ({cfg.num_seeds} seeds), bucket_size={bucket_size} steps/env ...")
    lowered = train_fn.lower(seeds, run_idxs)
    print("Compiling ...")
    lowered.compile()

    logger.start_time = time.time()
    jax.block_until_ready(jax.jit(jax.vmap(pre_eval))(seeds, run_idxs))
    ts, _ = train_fn(seeds, run_idxs)
    jax.block_until_ready(ts)

    # --- End-of-run summary tables from the final accumulators ---
    stats = _find_stats_state(ts.env_state)
    count = np.asarray(stats.count)[..., None] * np.ones_like(np.asarray(stats.mean))
    mean = np.asarray(stats.mean)
    m2 = np.asarray(stats.m2)
    flat = (
        count.reshape(-1, n_buckets, count.shape[-1]),
        mean.reshape(-1, n_buckets, mean.shape[-1]),
        m2.reshape(-1, n_buckets, m2.shape[-1]),
    )
    train_count, train_mean, train_m2 = _merge(*flat, axis=0)
    total_count, total_mean, total_m2 = _merge(
        train_count, train_mean, train_m2, axis=0
    )

    total_rows = _sensor_rows(
        layout, names, noise_cfg, total_count, total_mean, total_m2
    )
    _print_table("TOTAL - all training observations, all seeds", total_rows)

    logger.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))