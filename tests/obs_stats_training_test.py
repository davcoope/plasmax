"""Critic-accuracy metrics logged by obs_stats_training."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import TruncationWrapper

pytest.importorskip("rejax")

from helpers import CheapBoundaryEnv

from agents.ppo import PPOAdapter
from training.envelope_gymnax import EnvelopeGymnax
from training.obs_stats_training import _critic_metrics

GAMMA = 0.9


def _algo(**kwargs):
    env = TruncationWrapper(
        CheapBoundaryEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,)),
        max_steps=2,
    )
    gymnax_env = EnvelopeGymnax(env)
    algo = PPOAdapter.create(
        env=gymnax_env, env_params=gymnax_env.default_params, gamma=GAMMA, **kwargs
    )
    return algo, algo.init_state(jax.random.PRNGKey(0))


def _traj():
    """Two episodes in a 4-slot scan: one terminates after 3 steps (last slot
    is padding), one is truncated after 4."""
    reward = jnp.array([[1.0, 2.0, 3.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    valid = jnp.array([[True, True, True, False], [True, True, True, True]])
    terminated = jnp.array([[False, False, True, False], [False] * 4])
    truncated = jnp.array([[False] * 4, [False, False, False, True]])
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 2))
    return SimpleNamespace(
        reward=reward, valid=valid, terminated=terminated, truncated=truncated, obs=obs
    )


def _expected_targets(scale=1.0):
    g = GAMMA
    ep0 = [1 + 2 * g + 3 * g**2, 2 + 3 * g, 3.0]
    ep1 = [1 + g + g**2 + g**3, 1 + g + g**2, 1 + g, 1.0]
    return np.array(ep0 + ep1) / scale


def test_critic_metrics_match_hand_computed_returns():
    algo, ts = _algo()
    traj = _traj()
    metrics = _critic_metrics(algo, ts, traj)

    target = _expected_targets()
    value = np.asarray(
        algo.critic.apply(ts.critic_ts.params, traj.obs.reshape(-1, 2))
    ).reshape(2, 4)[np.asarray(traj.valid)]
    np.testing.assert_allclose(metrics["critic_eval/target_mean"], target.mean(), rtol=1e-5)
    np.testing.assert_allclose(metrics["critic_eval/value_mean"], value.mean(), rtol=1e-5)
    expected_ev = 1 - np.var(target - value) / np.var(target)
    np.testing.assert_allclose(
        metrics["critic_eval/explained_variance"], expected_ev, rtol=1e-4
    )


def test_critic_metrics_targets_use_reward_normaliser():
    algo, ts = _algo(normalize_rewards=True)
    ts = ts.replace(rew_rms_state=ts.rew_rms_state.replace(var=jnp.float32(4.0)))
    metrics = _critic_metrics(algo, ts, _traj())
    np.testing.assert_allclose(
        metrics["critic_eval/target_mean"], _expected_targets(scale=2.0).mean(), rtol=1e-5
    )
