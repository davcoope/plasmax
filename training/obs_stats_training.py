"""Observation signal statistics collected during real PPO training.

Trains iter/hybrid/flattop exactly as ``train_ppo.py`` does, but inserts an
``ObsStatsWrapper`` into the environment stack that accumulates per-sensor
running mean/variance (Welford) over every observation the agent actually
receives. The point is to measure each sensor's *natural* variation under a
genuinely learning policy, rather than under the zero-action or
random-action proxies used by ``experiments/studies/signal_to_noise_ratio``.

Training and evaluation statistics are kept separate without any explicit
flag: ``rejax``'s evaluator builds fresh env states via ``env.reset`` and
discards them, so accumulations made during evaluation never merge back
into the training state. Eval statistics are therefore computed separately,
from the eval rollout's own observations, inside the eval callback.

Three sets of numbers are produced. The number of buckets is not fixed -
it equals the number of eval checkpoints, ceil(total_timesteps / eval_freq),
so train bucket i and eval checkpoint i cover the same span of training:

* ``total``    - one mean/std per sensor over all training observations.
* ``train[i]`` - disjoint buckets, each the training window before eval i.
* ``eval[i]``  - one per eval checkpoint, eval rollouts only.

Statistics are taken in the *degraded* observation space (what the policy
actually sees: downsampled profiles, filtered channels, delay applied), not
the full-resolution space the noise wrapper operates on. Run with
``--env.noise-multiplier 0.0`` so that the measured variation is the
plasma's own, uncontaminated by injected observation noise.

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

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb
from envelope import WrappedState, Wrapper, field, static_field

from experiments.plotting.wandb_logging import _collect_returns_and_lengths
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
from scripts.project_paths import wandb_dir
from training.train_ppo import Config, _build_algo, _run_name

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
        bucket = jnp.minimum(state.step // self.bucket_size, self.n_buckets - 1)

        # Welford update, applied only to the active bucket's row.
        count_b = state.count[bucket] + 1.0
        delta = info.obs - state.mean[bucket]
        mean_b = state.mean[bucket] + delta / count_b
        m2_b = state.m2[bucket] + delta * (info.obs - mean_b)

        next_state = ObsStatsState(
            inner_state=inner_state,
            count=state.count.at[bucket].set(count_b),
            mean=state.mean.at[bucket].set(mean_b),
            m2=state.m2.at[bucket].set(m2_b),
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


def _make_eval_callback(num_steps: int, n_seeds: int, eval_rng, deterministic: bool):
    """Return (returns, lengths, eval obs stats) per eval checkpoint.

    Deliberately does no host-side logging: under ``vmap`` over seeds that
    needs buffering machinery, and everything here is logged once at the end
    from the stacked return values instead.
    """

    def _callback(algo, ts, rng, train_metrics=None):
        del train_metrics
        traj, returns, lengths = _collect_returns_and_lengths(
            algo,
            ts,
            eval_rng if eval_rng is not None else rng,
            num_steps,
            n_seeds,
            lean=True,
            deterministic=deterministic,
        )
        # traj.obs: (n_seeds, num_steps, obs_dim); traj.valid: (n_seeds, num_steps).
        valid = traj.valid[..., None]
        count = jnp.maximum(jnp.sum(traj.valid).astype(jnp.float32), 1.0)
        mean = jnp.sum(jnp.where(valid, traj.obs, 0.0), axis=(0, 1)) / count
        m2 = jnp.sum(jnp.where(valid, (traj.obs - mean) ** 2, 0.0), axis=(0, 1))
        return returns, lengths, (count, mean, m2)

    return _callback


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

    # Reuse train_ppo's builder so agent_kwargs/env_params/normalisation
    # options stay in sync with the launcher rather than drifting.
    algo = _build_algo(cfg, env)
    algo = algo.with_eval_callback(
        _make_eval_callback(
            num_steps=algo.env_params.max_steps_in_episode,
            n_seeds=cfg.env.eval_n_envs,
            eval_rng=jax.random.PRNGKey(cfg.env.eval_seed),
            deterministic=cfg.env.deterministic_eval,
        )
    )

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        mode=cfg.wandb.mode,
        name=f"{_run_name(cfg)}-obsstats",
        config=dataclasses.asdict(cfg),
        dir=str(wandb_dir()),
    )

    seeds = seed_keys(cfg.seed, cfg.num_seeds)
    train_fn = jax.jit(jax.vmap(algo.train))
    print(f"Training {cfg.num_seeds} seeds, bucket_size={bucket_size} steps/env ...")
    ts, evaluation = train_fn(seeds)
    jax.effects_barrier()

    # --- Training statistics: read the accumulators out of the final state ---
    stats = _find_stats_state(ts.env_state)
    # (n_seeds, num_envs, n_buckets, obs_dim) -> merge over seeds and envs.
    count = np.asarray(stats.count)[..., None] * np.ones_like(np.asarray(stats.mean))
    mean = np.asarray(stats.mean)
    m2 = np.asarray(stats.m2)
    flat = (count.reshape(-1, n_buckets, count.shape[-1]),
            mean.reshape(-1, n_buckets, mean.shape[-1]),
            m2.reshape(-1, n_buckets, m2.shape[-1]))
    train_count, train_mean, train_m2 = _merge(*flat, axis=0)
    total_count, total_mean, total_m2 = _merge(
        train_count, train_mean, train_m2, axis=0
    )

    # --- Eval statistics: stacked over checkpoints by rejax's scan ---
    # rejax prepends one extra, pre-training evaluation at index 0
    # (skip_initial_evaluation defaults False) - kept, but reported
    # separately below since it has no corresponding train bucket.
    eval_count, eval_mean, eval_m2 = (np.asarray(x) for x in evaluation[2])
    # (n_seeds, n_buckets + 1, ...) -> merge over seeds only.
    eval_count = eval_count[..., None] * np.ones_like(eval_mean)
    eval_count, eval_mean, eval_m2 = _merge(eval_count, eval_mean, eval_m2, axis=0)

    total_rows = _sensor_rows(
        layout, names, noise_cfg, total_count, total_mean, total_m2
    )
    _print_table("TOTAL (all training observations)", total_rows)

    for i in range(n_buckets):
        rows = _sensor_rows(
            layout, names, noise_cfg, train_count[i], train_mean[i], train_m2[i]
        )
        _print_table(f"TRAIN bucket {i + 1}/{n_buckets}", rows)
        for row in rows:
            wandb.log(
                {f"obs_train/{row['sensor']}/{k}": v for k, v in row.items() if k != "sensor"},
                step=int((i + 1) * cfg.ppo.eval_freq),
            )

    n_checkpoints = eval_mean.shape[0]
    for i in range(n_checkpoints):
        rows = _sensor_rows(
            layout, names, noise_cfg, eval_count[i], eval_mean[i], eval_m2[i]
        )
        _print_table(f"EVAL checkpoint {i + 1}/{n_checkpoints}", rows)
        for row in rows:
            wandb.log(
                {f"obs_eval/{row['sensor']}/{k}": v for k, v in row.items() if k != "sensor"},
                step=int((i + 1) * cfg.ppo.eval_freq),
            )

    table = wandb.Table(columns=[name for name, _, _ in _HEADERS])
    for row in total_rows:
        table.add_data(*[row[name] for name, _, _ in _HEADERS])
    wandb.log({"obs_total/table": table})

    print(f"\nreturns (final checkpoint): {np.asarray(evaluation[0])[:, -1].mean():.3f}")
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))