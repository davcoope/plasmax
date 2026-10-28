"""PPO with the reward's score scale calibrated from the initial policy.

Registered rewards return ``squareplus(s * x)`` for a raw score ``x``. Before
training, this script rolls out each seed's untrained policy and sets
``s = target / quantile(|x|, coverage)``: by default 90% of initial-policy
scores land in [-2, 2], where squareplus turns from its ~1/step survival floor
into its linear regime. Rewards are normalised by default. Passing
``--env.reward-score-scale`` skips calibration and uses that value instead.

Training otherwise matches ``train_ppo.py``'s vmapped path (used for
``--num-seeds 1`` too, so there are no single-seed Plotly figures), with extra
eval-rollout metrics: the raw score return (comparable across ``s``), applied
actuators, critic accuracy, and whether the scaled scores are still O(1).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb

from experiments.studies.baseline_study import seed_keys
from plasmax.wrappers import unwrap_to_env_state
from training import train_ppo, wandb_logging
from training.evaluation import write_transfer_summary
from training.runs import evaluate_transfer, save_run_policies, validate_seeds
from training.train_ppo import (
    Config,
    EnvConfig,
    PPOConfig,
    _backend_kind,
    _build_algo,
    _load_env,
    _run_name,
    _timed,
)
from training.vmap_logging import SeedBufferLogger
from training.wandb_logging import (
    _collect_returns_and_lengths,
    _masked_mean,
    evaluation_scalar_metrics,
)

# ---------------------------------------------------------------------------
# Upstream drift check
# ---------------------------------------------------------------------------

# _run and _make_callback are adapted copies of these two functions (train_ppo
# has no hook for metrics computed from the eval rollouts). If either changes
# upstream, port the change into the copy, then update the hash to the one the
# warning prints.
_UPSTREAM = (train_ppo._run_vmap, wandb_logging.make_buffered_seed_callback)
_UPSTREAM_SHA256 = "52b32fd13a15da7b6ef5dbddd66b2d2d9d325ca2f40e9aa59dff077366a49aac"


def _check_upstream() -> None:
    source = "".join(inspect.getsource(fn) for fn in _UPSTREAM)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != _UPSTREAM_SHA256:
        print(
            "warning: train_ppo._run_vmap or "
            "wandb_logging.make_buffered_seed_callback has changed since "
            "auto_scale_ppo._run / _make_callback were adapted from them. Port "
            "any relevant change, then set "
            f"_UPSTREAM_SHA256 = {digest!r}."
        )


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AutoScaleEnvConfig(EnvConfig):
    # None = calibrate from the initial policy; a value skips calibration.
    reward_score_scale: float | None = None


@dataclasses.dataclass
class CalibConfig:
    # |s * score| at the coverage quantile of initial-policy scores.
    target: float = 2.0
    # Fraction of initial-policy scores scaled into [-target, target].
    coverage: float = 0.9
    # Initial-policy episodes per seed.
    n_envs: int = 32


@dataclasses.dataclass
class AutoScaleConfig(Config):
    env: AutoScaleEnvConfig = dataclasses.field(default_factory=AutoScaleEnvConfig)
    ppo: PPOConfig = dataclasses.field(
        default_factory=lambda: PPOConfig(normalize_rewards=True)
    )
    calib: CalibConfig = dataclasses.field(default_factory=CalibConfig)


class _Label(str):
    """A string that formats as itself under any spec, e.g. ``f"{x:g}"``."""

    def __format__(self, spec: str) -> str:
        return str(self)


def _auto_run_name(cfg: AutoScaleConfig) -> str:
    """train_ppo's run name with ``-rsauto`` in place of ``-rs{s}``."""
    label = "auto"
    defaults = CalibConfig()
    if cfg.calib.target != defaults.target:
        label += f"-t{cfg.calib.target:g}"
    if cfg.calib.coverage != defaults.coverage:
        label += f"-c{cfg.calib.coverage:g}"
    env = dataclasses.replace(cfg.env, reward_score_scale=_Label(label))
    return _run_name(dataclasses.replace(cfg, env=env))


# ---------------------------------------------------------------------------
# Raw scores and calibration
# ---------------------------------------------------------------------------


def _score_fn(env):
    """The env's own reward (with any scenario binding, e.g. lh_transition's
    t_final) reduced to its raw score: no squareplus, unscaled, zero on
    termination."""
    return functools.partial(
        env.unwrapped._dynamics._reward_fn, squareplus=False, scale=1.0
    )


def _raw_scores(traj, score_fn):
    """Raw score of every (episode, step) of a collected trajectory."""
    env_state = unwrap_to_env_state(traj.env_state)
    # Registered rewards score only the post-step state.
    return jax.vmap(jax.vmap(lambda s, c: score_fn(None, None, s, c)))(
        env_state, traj.info.termination_code
    )


def _calibrate(cfg: AutoScaleConfig, env, score_fn) -> tuple[float, dict]:
    """``(s, summary)`` from rollouts of every seed's untrained policy.

    Uses the scores of ordinary (non-terminal) steps, pooled over seeds: the
    vmapped seeds share one env, so they share one ``s``.
    """
    algo = _build_algo(cfg, env)
    num_steps = algo.env_params.max_steps_in_episode

    def rollout(rng):
        # The same initial policy algo.train(rng) starts from.
        ts = algo.init_state(rng)
        traj, _, _ = _collect_returns_and_lengths(
            algo, ts, ts.rng, num_steps, cfg.calib.n_envs, lean=True
        )
        ordinary = traj.valid & (traj.info.termination_code == -1)
        return _raw_scores(traj, score_fn), ordinary

    scores, ordinary = jax.jit(jax.vmap(rollout))(seed_keys(cfg.seed, cfg.num_seeds))
    scores = np.asarray(scores)[np.asarray(ordinary)]
    if scores.size == 0:
        raise RuntimeError("calibration rollouts produced no non-terminal steps")
    quantile = float(np.quantile(np.abs(scores), cfg.calib.coverage))
    if not quantile > 0.0:
        raise RuntimeError(
            f"the {cfg.calib.coverage:g} quantile of |score| is {quantile}; "
            "cannot calibrate a scale from it"
        )
    scale = cfg.calib.target / quantile
    p05, p50, p95 = np.percentile(scores, [5, 50, 95])
    summary = {
        "calib/score_scale": scale,
        "calib/abs_score_quantile": quantile,
        "calib/n_scores": int(scores.size),
        "calib/score_p05": float(p05),
        "calib/score_p50": float(p50),
        "calib/score_p95": float(p95),
    }
    return scale, summary


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


def _physics_metrics(traj, score, actuator_names, score_name):
    """Eval score return and mean applied actuators (MW).

    Score return is the per-episode sum of the reward's raw score (no
    squareplus, unscaled, zero on termination), averaged over eval episodes.
    """
    score_return = jnp.sum(jnp.where(traj.valid, score, 0.0), axis=1).mean()
    metrics = {f"physics_eval/{score_name}_return": score_return}
    env_state = unwrap_to_env_state(traj.env_state)
    applied = _masked_mean(env_state.prev_action, traj.valid)
    for i, name in enumerate(actuator_names):
        if name.startswith("P_"):
            metrics[f"actuators_eval/{name}_MW"] = applied[i] * 1e-6
        else:
            metrics[f"actuators_eval/{name}"] = applied[i]
    return metrics


def _calib_metrics(traj, score, scale, target, coverage):
    """Whether the scaled scores are still O(1), as calibration made them.

    Over ordinary (non-terminal) eval steps: ``frac_in_range`` is the fraction
    with ``|s * x| <= target`` (``coverage`` at calibration), and
    ``abs_scaled_score_quantile`` is the ``coverage`` quantile of ``|s * x|``
    (``target`` at calibration).
    """
    ordinary = traj.valid & (traj.info.termination_code == -1)
    scaled = jnp.abs(scale * score)
    count = jnp.maximum(jnp.sum(ordinary), 1)
    return {
        "calib_eval/frac_in_range": jnp.sum(ordinary & (scaled <= target)) / count,
        "calib_eval/abs_scaled_score_quantile": jnp.nanquantile(
            jnp.where(ordinary, scaled, jnp.nan), coverage
        ),
    }


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

    def mean(x):
        return jnp.sum(jnp.where(mask, x, 0.0)) / count

    def var(x):
        return mean((x - mean(x)) ** 2)

    target_var = var(target)
    return {
        "critic_eval/value_mean": mean(value),
        "critic_eval/target_mean": mean(target),
        "critic_eval/explained_variance": 1.0
        - var(target - value) / jnp.maximum(target_var, 1e-12),
    }


# ---------------------------------------------------------------------------
# Eval callback and logger
# ---------------------------------------------------------------------------


def _make_callback(
    logger,
    *,
    score_fn,
    score_name,
    actuator_names,
    scale,
    target,
    coverage,
    num_steps,
    n_seeds,
    eval_rng,
    deterministic,
):
    """``run_idx -> callback`` factory, adapted from make_buffered_seed_callback
    (physics kind) with the extra metrics computed from the same rollouts."""

    def for_run(run_idx):
        def _callback(algo, ts, rng, train_metrics):
            traj, episode_returns, episode_lengths = _collect_returns_and_lengths(
                algo,
                ts,
                eval_rng,
                num_steps,
                n_seeds,
                lean=True,
                deterministic=deterministic,
            )
            metrics = evaluation_scalar_metrics(
                traj, episode_returns, episode_lengths, train_metrics, physics=True
            )
            score = _raw_scores(traj, score_fn)
            metrics.update(_physics_metrics(traj, score, actuator_names, score_name))
            metrics.update(_calib_metrics(traj, score, scale, target, coverage))
            metrics.update(_critic_metrics(algo, ts, traj))
            jax.debug.callback(logger.log, ts.global_step, run_idx, metrics)
            return episode_returns, episode_lengths

        return _callback

    return for_run


class _EvalLogger(SeedBufferLogger):
    """SeedBufferLogger that also prints eval score and critic lines at flush time.

    Printing here (rather than from a separate debug callback) keeps the
    lines adjacent to their own ``step=`` progress line instead of racing it -
    ``jax.debug.callback`` ordering across a vmapped seed axis is not
    guaranteed.
    """

    def __init__(self, *, score_name: str, target: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.score_name = score_name
        self.target = target

    def _flush_step(self, step: int) -> None:
        # Read the buffer before super() pops it, so the lines print above
        # the step= line that super() emits.
        per_seed = self._buffers.get(step, {})
        if per_seed:
            runs = sorted(per_seed)

            def seeds(key):
                return np.asarray([per_seed[r][key] for r in runs])

            score = seeds(f"physics_eval/{self.score_name}_return")
            length = seeds("evaluation/episode_length_mean")
            spread = float(np.std(score, ddof=1)) if score.size > 1 else 0.0
            in_range = np.mean(seeds("calib_eval/frac_in_range"))
            self.print_fn(
                f"EVAL {self.score_name} return = {np.mean(score):.4g} "
                f"±{spread:.2g} (mean {np.mean(score / length):.4g}/step; "
                f"across seeds), |s·x| <= {self.target:g} on {in_range:.0%} "
                "of steps"
            )
            self.print_fn(
                f"EVAL critic: V mean = {np.mean(seeds('critic_eval/value_mean')):.4g} "
                f"vs target mean = {np.mean(seeds('critic_eval/target_mean')):.4g}, "
                "explained variance = "
                f"{np.mean(seeds('critic_eval/explained_variance')):.3f} "
                "(across seeds)"
            )
        super()._flush_step(step)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _run(cfg: AutoScaleConfig, run_name: str, calib_summary: dict) -> None:
    """Adapted from train_ppo._run_vmap: the eval callback and logger above,
    and ``calib_summary`` logged up front and saved with the policies."""
    env = _load_env(cfg, cfg.env.backend)
    score_name = cfg.env.reward or env.plasmax_config.task.reward
    logger = _EvalLogger(
        score_name=score_name,
        target=cfg.calib.target,
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
    if calib_summary:
        logger.log_once(calib_summary)

    algo = _build_algo(cfg, env)
    for_run = _make_callback(
        logger,
        score_fn=_score_fn(env),
        score_name=score_name,
        actuator_names=[spec.name for spec in env.actuator_specs],
        scale=cfg.env.reward_score_scale,
        target=cfg.calib.target,
        coverage=cfg.calib.coverage,
        num_steps=algo.env_params.max_steps_in_episode,
        n_seeds=cfg.env.eval_n_envs,
        eval_rng=jax.random.PRNGKey(cfg.env.eval_seed),
        deterministic=cfg.env.deterministic_eval,
    )

    # rejax evaluates once before training, but from outside its lax.scan -
    # with no data dependency on the scan, XLA is free to schedule that
    # callback after later ones, so W&B drops it as an out-of-order step. Run
    # it explicitly first instead, and tell rejax to skip its own.
    algo = algo.replace(skip_initial_evaluation=True)

    def pre_eval(rng, run_idx):
        algo_i = algo.with_eval_callback(for_run(run_idx))
        ts0 = algo_i.init_state(rng)
        return algo_i.eval_callback(algo_i, ts0, ts0.rng)

    def train_one(rng, run_idx):
        # Rebuild the callback inside the trace so each vmapped seed's run_idx
        # tracer flows through to logger.log via jax.debug.callback.
        algo_i = algo.with_eval_callback(for_run(run_idx))
        return algo_i.train(rng)

    seeds = seed_keys(cfg.seed, cfg.num_seeds)
    run_idxs = jnp.arange(cfg.num_seeds)
    train_fn = jax.jit(jax.vmap(train_one))

    lowered, t_lower = _timed(
        f"Lowering ({cfg.num_seeds} seeds)", lambda: train_fn.lower(seeds, run_idxs)
    )
    _, t_compile = _timed("Compiling", lowered.compile)

    logger.start_time = time.time()

    def _train():
        jax.block_until_ready(jax.jit(jax.vmap(pre_eval))(seeds, run_idxs))
        ts, results = train_fn(seeds, run_idxs)
        jax.block_until_ready((ts, results))
        jax.effects_barrier()
        return ts, results

    (ts, results), t_train = _timed("Training", _train)

    log_once = {
        "time/lower_s": t_lower,
        "time/compile_s": t_compile,
        "time/train_s": t_train,
        "run/actual_train_steps": int(np.asarray(ts.global_step[0])),
    }

    if cfg.env.transfer_backend is not None:
        transfer_summary = evaluate_transfer(
            algo,
            ts,
            env,
            _load_env(cfg, cfg.env.transfer_backend),
            cfg,
            batched=True,
        )
        log_once.update(transfer_summary)
        write_transfer_summary(cfg.history_dir, run_name, transfer_summary)

    if cfg.save_policy:
        checkpoints = save_run_policies(
            algo,
            ts,
            cfg,
            run_name,
            batched=True,
            results=results,
            metrics={**calib_summary, **log_once},
        )
        if checkpoints:
            artifact = wandb.Artifact(run_name, type="model")
            for checkpoint in checkpoints:
                artifact.add_file(str(checkpoint))
                print(f"Saved checkpoint to {checkpoint}", flush=True)
            logger.log_artifact(artifact)

    logger.log_once(log_once)
    logger.finish()
    print("Done.")


def main(cfg: AutoScaleConfig) -> None:
    _check_upstream()
    validate_seeds(cfg.env.backend, cfg.num_seeds)
    if _backend_kind(cfg.env.env_setup) == "world_model":
        raise ValueError(
            "auto_scale_ppo does not work with world-model envs: it scales a "
            "registered reward's score, and world models use their native "
            "reward. Use training/train_ppo.py instead."
        )

    if cfg.env.reward_score_scale is not None:
        _run(cfg, cfg.run_name or _run_name(cfg), {})
        return

    if not cfg.env.squareplus:
        raise ValueError(
            "--env.no-squareplus needs an explicit --env.reward-score-scale: "
            "calibration targets squareplus's transition region"
        )
    run_name = cfg.run_name or _auto_run_name(cfg)
    unscaled = dataclasses.replace(
        cfg, env=dataclasses.replace(cfg.env, reward_score_scale=1.0)
    )
    env = _load_env(unscaled, cfg.env.backend)
    (scale, summary), t_calib = _timed(
        "Calibrating reward score scale",
        lambda: _calibrate(unscaled, env, _score_fn(env)),
    )
    summary["time/calib_s"] = t_calib
    # Recorded in the W&B config and saved policy metadata, so
    # load_policy_env rebuilds the reward with the calibrated scale.
    cfg = dataclasses.replace(
        cfg, env=dataclasses.replace(cfg.env, reward_score_scale=scale)
    )
    _run(cfg, run_name, summary)


if __name__ == "__main__":
    main(tyro.cli(AutoScaleConfig))
