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
import wandb
from envelope import WrappedState, Wrapper, field, static_field

from training.wandb_logging import (
    _collect_returns_and_lengths,
    _masked_mean,
    evaluation_scalar_metrics,
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
    unwrap_to_env_state,
)
from training.runs import save_run_policies
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
        squareplus=cfg.env.squareplus,
        reward_scale=cfg.env.reward_score_scale,
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


def _physics_metrics(traj, actuator_names):
    """Eval P_diff return (GW summed over steps) and mean applied actuators (MW).

    P_diff return is the per-episode sum of unscaled P_diff, averaged over eval
    episodes.
    """
    env_state = unwrap_to_env_state(traj.env_state)
    plasma = env_state.plasma
    live = traj.info.termination_code == -1
    p_diff = jnp.where(live, (plasma.P_fusion - plasma.P_aux_total) * 1e-9, 0.0)
    p_diff_return = jnp.sum(jnp.where(traj.valid, p_diff, 0.0), axis=1).mean()
    metrics = {"physics_eval/P_diff_return": p_diff_return}
    applied = _masked_mean(env_state.prev_action, traj.valid)
    for i, name in enumerate(actuator_names):
        if name.startswith("P_"):
            metrics[f"actuators_eval/{name}_MW"] = applied[i] * 1e-6
        else:
            metrics[f"actuators_eval/{name}"] = applied[i]
    return metrics


def _critic_metrics(algo, ts, traj):
    """How well the critic predicts what actually happened in the eval rollouts.

    - ``value_mean`` / ``target_mean``: whether the critic's *level* is right.
    - ``explained_variance`` = 1 - Var(target - V) / Var(target): whether its
      *shape* is. 1 = tracks every state-to-state difference, 0 = no better
      than predicting one constant, < 0 = its differences are wrong.
    """
    reward = traj.reward
    if algo.normalize_rewards:
        reward = algo.normalize_rew(ts.rew_rms_state, reward)
    reward = jnp.where(traj.valid, reward, 0.0)
    ended = traj.terminated | traj.truncated

    def discounted(future, step):
        r, end = step
        g = r + algo.gamma * jnp.where(end, 0.0, future)
        return g, g

    # Scan backwards over time (axis 1); the carry is one return per episode.
    _, target = jax.lax.scan(
        discounted,
        jnp.zeros(reward.shape[0]),
        (reward.T, ended.T),
        reverse=True,
    )
    target = target.T

    obs = traj.obs
    if algo.normalize_observations:
        obs = algo.normalize_obs(ts.obs_rms_state, obs)
    flat = obs.reshape(-1, obs.shape[-1])
    value = algo.critic.apply(ts.critic_ts.params, flat).reshape(target.shape)

    mask = traj.valid
    count = jnp.maximum(jnp.sum(mask), 1)
    mean = lambda x: jnp.sum(jnp.where(mask, x, 0.0)) / count
    var = lambda x: mean((x - mean(x)) ** 2)
    target_var = var(target)
    return {
        "critic_eval/value_mean": mean(value),
        "critic_eval/target_mean": mean(target),
        "critic_eval/explained_variance": 1.0
        - var(target - value) / jnp.maximum(target_var, 1e-12),
    }


# ---------------------------------------------------------------------------
# Eval callback
# ---------------------------------------------------------------------------


def _make_logging_callback(
    logger,
    *,
    layout,
    names,
    noise_cfg,
    actuator_names,
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
    own eval rollouts (``obs_eval/*``, computed from the eval trajectory),
    and the eval rollouts' true P_diff and applied actuator means
    (``physics_eval/*``, ``actuators_eval/*``; see :func:`_physics_metrics`)
    and critic accuracy on them (``critic_eval/*``; see :func:`_critic_metrics`).
    Routing through ``logger.log`` via ``jax.debug.callback`` means it
    appears live during training rather than only at the end. Each seed
    reports its own statistics (including its own SNR); the logger then
    reports the mean and spread across seeds, as it does for ``return``.
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
            metrics = evaluation_scalar_metrics(
                traj, episode_returns, episode_lengths, train_metrics, physics=True
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
            metrics.update(_physics_metrics(traj, actuator_names))
            metrics.update(_critic_metrics(algo, ts, traj))
            jax.debug.callback(logger.log, ts.global_step, run_idx, metrics)
            return episode_returns, episode_lengths

        return _callback

    return for_run


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class _PDiffLogger(SeedBufferLogger):
    """SeedBufferLogger that also prints eval P_diff and critic lines at flush time.

    Printing here (rather than from a separate debug callback) keeps the
    line adjacent to its own ``step=`` progress line instead of racing it -
    ``jax.debug.callback`` ordering across a vmapped seed axis is not
    guaranteed. Per-sensor observation statistics go to W&B only.
    """

    def _flush_step(self, step: int) -> None:
        # Read the buffer before super() pops it, so the line prints above
        # the step= line that super() emits.
        per_seed = self._buffers.get(step, {})
        if per_seed:
            runs = sorted(per_seed)
            p_diff = np.asarray([per_seed[r]["physics_eval/P_diff_return"] for r in runs])
            length = np.asarray(
                [per_seed[r]["evaluation/episode_length_mean"] for r in runs]
            )
            spread = float(np.std(p_diff, ddof=1)) if p_diff.size > 1 else 0.0
            print(
                f"EVAL P_diff return = {np.mean(p_diff):.1f} ±{spread:.1f} GW·steps "
                f"(mean {np.mean(p_diff / length):.4f} GW/step; across seeds)"
            )
            critic = {
                k: np.asarray([per_seed[r][f"critic_eval/{k}"] for r in runs])
                for k in ("value_mean", "target_mean", "explained_variance")
            }
            print(
                f"EVAL critic: V mean = {np.mean(critic['value_mean']):.4g} vs "
                f"target mean = {np.mean(critic['target_mean']):.4g}, "
                f"explained variance = {np.mean(critic['explained_variance']):.3f} "
                "(across seeds)"
            )
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
    logger = _PDiffLogger(
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
    )

    # Reuse train_ppo's builder so agent_kwargs/env_params/normalisation
    # options stay in sync with the launcher rather than drifting.
    algo = _build_algo(cfg, env)
    for_run = _make_logging_callback(
        logger,
        layout=layout,
        names=names,
        noise_cfg=noise_cfg,
        actuator_names=[spec.name for spec in env.actuator_specs],
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
    ts, results = train_fn(seeds, run_idxs)
    jax.block_until_ready((ts, results))
    jax.effects_barrier()

    if cfg.save_policy:
        checkpoints = save_run_policies(
            algo, ts, cfg, run_name, batched=True, results=results
        )
        if checkpoints:
            artifact = wandb.Artifact(run_name, type="model")
            for checkpoint in checkpoints:
                artifact.add_file(str(checkpoint))
                print(f"Saved checkpoint to {checkpoint}", flush=True)
            logger.log_artifact(artifact)

    logger.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))