"""Observation statistics collected during real PPO training.

Trains iter/hybrid/flattop like ``train_ppo.py``, with an ``ObsStatsWrapper``
that tracks running per-sensor mean/variance (Welford) of every observation
the agent receives - measuring natural variation under a real, learning
policy, unlike the zero/random-action proxies in
``experiments/studies/signal_to_noise_ratio``.
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

    Stats are calculated separately for each environment. ``init`` starts the
    accumulators at zero. ``reset`` is called when an episode ends and a new
    one begins - it ensures that auto-reset does not wipe stats.

    ``m2`` is the sum of squared deviations from the running mean (Welford).
    Variance is ``m2 / count``.
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
        # An invalid-state disruption (core.py's termination_code=4) emits a
        # non-finite obs. Skipping it keeps one NaN from poisoning the whole
        # bucket, since Welford's recurrence has no recovery path.
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

    Shapes are ``(..., N, ...)`` with ``N`` the axis being reduced. Entries
    with zero samples contribute nothing and are guarded against division by
    zero.
    """
    total = jnp.sum(count, axis=axis)
    safe = jnp.where(total == 0.0, 1.0, total)
    merged_mean = jnp.sum(count * mean, axis=axis) / safe
    spread = (mean - jnp.expand_dims(merged_mean, axis)) ** 2
    merged_m2 = jnp.sum(m2 + count * spread, axis=axis)
    return total, merged_mean, merged_m2


def _obs_metrics(prefix, layout, names, noise_cfg, count, mean, m2):
    """Flat ``{metric_name: scalar}`` dict of one seed's bucket statistics.

    ``count``/``mean``/``m2`` are per-observation-channel arrays of shape
    ``(obs_dim,)``. Channels belonging to one sensor are merged so each
    sensor contributes a single rms/std/snr.

    ``NoiseWrapper`` draws ``noise_t = rel_std * eps_t * obs_t`` with
    ``eps_t`` zero-mean, unit-variance and independent of ``obs_t``, so
    ``std(noise) = rel_std * sqrt(E[obs^2]) = rel_std * RMS(obs)`` - the
    mean alone understates it. Pooled over a sensor's channels,
    ``RMS^2 = mean^2 + std^2``.

    SNR is kept as signal-over-noise and its reciprocal is never formed:
    since ``1/x`` is convex, averaging ``noise_std / obs_std`` across seeds
    would exceed the reciprocal of the averaged SNR (Jensen's inequality),
    overstating noise dominance whenever seeds disagree.
    """
    metrics = {}
    for name in names:
        sl = layout.slice_of(name)
        total, sensor_mean, sensor_m2 = _merge(count[sl], mean[sl], m2[sl], axis=0)
        safe = jnp.where(total == 0.0, 1.0, total)
        obs_std = jnp.sqrt(jnp.maximum(sensor_m2 / safe, 0.0))
        obs_rms = jnp.sqrt(sensor_mean**2 + obs_std**2)
        noise_std = float(noise_cfg.get(name, 0.0)) * obs_rms
        metrics[f"{prefix}/{name}/obs_rms"] = obs_rms
        metrics[f"{prefix}/{name}/obs_std"] = obs_std
        metrics[f"{prefix}/{name}/noise_std_at_1x"] = noise_std
        # Infinite for a sensor with no wrappers.yaml noise entry (``t``).
        metrics[f"{prefix}/{name}/snr"] = obs_std / noise_std
    return metrics


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def _build_env(cfg: Config, bucket_size: int, n_buckets: int):
    """RealisticWrappers, plus the statistics wrapper.

    Mirrors ``plasmax.wrappers.RealisticWrappers`` exactly, with one
    addition: ObsStatsWrapper is inserted after the observation degradations
    (so it sees the degraded observation) but before the training wrappers
    (so it is unaffected by action rescaling). That mid-stack injection is
    why the composition is repeated here rather than delegated; keep this in
    step with RealisticWrappers when pulling upstream.
    """
    env = make(
        cfg.env.env_setup,
        cfg.env.backend,
        reward=cfg.env.reward,
    )
    plasmax_cfg = env.plasmax_config
    if plasmax_cfg.physics_randomization:
        env = PhysicsRandomizationWrapper(env)

    real = plasmax_cfg.observations.realistic
    if real.noise:
        env = NoiseWrapper(env, noise_multiplier=cfg.env.noise_multiplier)
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
    appears live during training rather than only at the end. Each seed
    reports its own statistics; :class:`_TableLogger` combines them across
    seeds at flush time, which also keeps each table ordered with its
    ``step=`` progress line.
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
            train_count, train_mean, train_m2 = _merge(
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


_HEADERS = (
    ("sensor", "<18", "s"),
    ("noise_rel_std", ">14", ".3f"),
    ("obs_rms", ">12", ".4g"),
    ("obs_std", ">12", ".4g"),
    ("noise_std_at_1x", ">16", ".4g"),
    ("snr_mean", ">12", ".3f"),
    ("snr_std", ">12", ".3f"),
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

    Each seed is an independently trained policy, so its SNR is computed in
    full before any cross-seed aggregation, and the table reports the mean
    and spread of those per-seed SNRs - as ``return`` is reported, rather
    than pooled as if the seeds were one population. The descriptive
    columns beside it are plain seed means, so ``obs_std / noise_std`` as
    printed will not exactly reproduce ``snr_mean``.

    Printing here (rather than from a separate debug callback) keeps each
    table adjacent to its own ``step=`` progress line instead of racing it -
    ``jax.debug.callback`` ordering across a vmapped seed axis is not
    guaranteed.
    """

    def __init__(self, *args, names, noise_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self._names = names
        self._noise_cfg = noise_cfg

    def _rows(self, prefix, per_seed):
        order = sorted(per_seed)

        def seed_values(key):
            return np.asarray([per_seed[r][key] for r in order], dtype=np.float64)

        rows = []
        for name in self._names:
            key = f"{prefix}/{name}"
            if f"{key}/snr" not in per_seed[order[0]]:
                return None
            snr = seed_values(f"{key}/snr")
            with np.errstate(invalid="ignore"):
                # nan once a seed is infinite, i.e. a sensor carrying no noise.
                snr_std = float(np.std(snr, ddof=1)) if snr.size > 1 else float("nan")
            rows.append(
                {
                    "sensor": name,
                    "noise_rel_std": float(self._noise_cfg.get(name, 0.0)),
                    "obs_rms": float(np.mean(seed_values(f"{key}/obs_rms"))),
                    "obs_std": float(np.mean(seed_values(f"{key}/obs_std"))),
                    "noise_std_at_1x": float(
                        np.mean(seed_values(f"{key}/noise_std_at_1x"))
                    ),
                    "snr_mean": float(np.mean(snr)),
                    "snr_std": snr_std,
                }
            )
        return rows

    def _flush_step(self, step: int) -> None:
        # Read the buffer before super() pops it, so the tables print above
        # the step= line that super() emits.
        per_seed = self._buffers.get(step, {})
        if per_seed:
            elapsed = time.time() - (self.start_time or time.time())
            # At step 0 no training bucket has closed, so only eval is real.
            if step > 0:
                rows = self._rows("obs_train", per_seed)
                if rows:
                    _print_table(f"TRAIN bucket @ step {step:,}", rows)
            rows = self._rows("obs_eval", per_seed)
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

    logger.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))