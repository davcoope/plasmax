"""PPO thin-adapter and end-to-end training smoke tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tyro
from envelope import Environment, TruncationWrapper, static_field
from flax import linen as nn

from plasmax.wrappers import RealisticWrappers

pytest.importorskip("rejax")

from helpers import CheapBoundaryEnv, CheapBoundaryState
from rejax.algos.ppo import PPO, AdvantageMinibatch, Trajectory
from rejax.networks import GaussianPolicy

from agents.ppo import MultiDiscretePolicy, PPOAdapter
from plasmax.environment.factory import make
from plasmax.wrappers import ActionRescaleWrapper, QuantizeActionWrapper
from training.envelope_gymnax import EnvelopeGymnax


def _make_algo(env: Environment, **ppo_kwargs) -> PPOAdapter:
    gymnax_env = EnvelopeGymnax(env)
    return PPOAdapter.create(
        env=gymnax_env,
        env_params=gymnax_env.default_params,
        **ppo_kwargs,
    )


def test_multidiscrete_policy_samples_and_scores_each_head():
    policy = MultiDiscretePolicy(
        nvec=(2, 4), hidden_layer_sizes=(8,), activation=nn.tanh
    )
    obs = jnp.zeros((3, 2), jnp.float32)
    params = policy.init(jax.random.PRNGKey(0), obs, jax.random.PRNGKey(1))
    actions, log_prob, entropy = policy.apply(params, obs, jax.random.PRNGKey(2))
    rescored_log_prob, rescored_entropy = policy.apply(
        params, obs, actions, method="log_prob_entropy"
    )

    assert actions.shape == (3, 2)
    assert jnp.all((actions[:, 0] >= 0) & (actions[:, 0] < 2))
    assert jnp.all((actions[:, 1] >= 0) & (actions[:, 1] < 4))
    assert jnp.all(jnp.isfinite(log_prob))
    assert jnp.all(jnp.isfinite(entropy))
    np.testing.assert_allclose(log_prob, rescored_log_prob, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(entropy, rescored_entropy, rtol=1e-5, atol=1e-8)


def test_adapter_uses_upstream_ppo_optimization():
    assert issubclass(PPOAdapter, PPO)
    assert "calculate_gae" not in PPOAdapter.__dict__
    assert "update" not in PPOAdapter.__dict__


def _critic_step(algo: PPOAdapter, moved: float, target_gap: float = 5.0):
    """Critic params after one update. The rollout's stored values sit
    ``moved`` below the current predictions (as if earlier minibatches already
    moved them) and the targets sit ``target_gap`` above the predictions."""
    ts = algo.init_state(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (16, 2))
    value = algo.critic.apply(ts.critic_ts.params, obs)
    zeros = jnp.zeros(16)
    traj = Trajectory(obs, jnp.zeros((16, 1)), zeros, zeros, value - moved, zeros)
    batch = AdvantageMinibatch(traj, zeros, value + target_gap)
    return ts.critic_ts.params, algo.update_critic(ts, batch).critic_ts.params


@pytest.mark.parametrize("moved", [0.05, 1.0], ids=("inside", "outside"))
def test_value_clip_default_matches_upstream_rejax(moved):
    env = TruncationWrapper(
        CheapBoundaryEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,)), max_steps=2
    )
    gymnax_env = EnvelopeGymnax(env)
    kwargs = dict(
        env=gymnax_env,
        env_params=gymnax_env.default_params,
        agent_kwargs={"activation": "swish", "hidden_layer_sizes": (64, 64)},
    )
    _, ours = _critic_step(PPOAdapter.create(**kwargs), moved)
    _, upstream = _critic_step(PPO.create(**kwargs), moved)
    jax.tree.map(np.testing.assert_allclose, ours, upstream)


def test_value_clip_none_is_plain_squared_error():
    env = TruncationWrapper(
        CheapBoundaryEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,)), max_steps=2
    )
    clipped = _make_algo(env, value_clip_eps=0.2)
    unclipped = _make_algo(env, value_clip_eps=None)
    huge_clip = _make_algo(env, value_clip_eps=1e9)
    # Predictions already moved 1.0 past their stored values, beyond a 0.2
    # clip: the clipped loss has zero gradient so the critic stays put, while
    # no clip (or an unreachable one) keeps moving towards the targets.
    before, p_clip = _critic_step(clipped, moved=1.0)
    _, p_none = _critic_step(unclipped, moved=1.0)
    _, p_huge = _critic_step(huge_clip, moved=1.0)
    jax.tree.map(np.testing.assert_allclose, p_clip, before)
    jax.tree.map(np.testing.assert_allclose, p_none, p_huge)
    leaves = lambda p: jnp.concatenate([x.ravel() for x in jax.tree.leaves(p)])
    assert not np.allclose(leaves(p_none), leaves(before))


@pytest.mark.parametrize(
    ("args", "expected"),
    [([], 0.2), (["--ppo.value-clip-eps", "None"], None), (["--ppo.value-clip-eps", "10"], 10.0)],
)
def test_launcher_parses_value_clip_eps(args, expected) -> None:
    from training import train_ppo

    cfg = tyro.cli(train_ppo.Config, args=args)
    assert cfg.ppo.value_clip_eps == expected
    suffix = {0.2: None, None: "-novclip", 10.0: "-vclip10"}[expected]
    name = train_ppo._run_name(cfg)
    assert (suffix in name) if suffix else ("vclip" not in name)


@pytest.mark.parametrize(
    "args", [["--ppo.residual-policy"], ["--ppo.initial-log-std", "-1"]]
)
def test_launcher_rejects_removed_policy_options(args: list[str]) -> None:
    from training import train_ppo

    with pytest.raises(SystemExit) as error:
        tyro.cli(train_ppo.Config, args=args)
    assert error.value.code != 0


class ResetActionState(CheapBoundaryState):
    prev_action: jax.Array


class ResetActionEnv(CheapBoundaryEnv):
    initial_action: tuple[float, ...] = static_field(default=(0.3,))

    def init(self, key):
        del key
        state = ResetActionState(
            obs=jnp.zeros((self.obs_dim,), jnp.float32),
            steps=jnp.asarray(0, jnp.int32),
            prev_action=jnp.asarray(self.initial_action, jnp.float32),
        )
        return state, self._info(state)


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_actions_do_not_depend_on_reset_actuator_defaults(
    rescale: bool, deterministic: bool
) -> None:
    actions = []
    observation = jnp.asarray([0.2, -0.4], jnp.float32)
    for default in (-1.5, 2.5):
        env = ResetActionEnv(
            obs_dim=2,
            action_low=(-2.0,),
            action_high=(3.0,),
            initial_action=(default,),
        )
        if rescale:
            env = ActionRescaleWrapper(env)
        algo = _make_algo(
            TruncationWrapper(env, max_steps=2),
            num_envs=1,
            agent_kwargs={"hidden_layer_sizes": (4,)},
        )
        assert type(algo.actor) is GaussianPolicy
        state = algo.init_state(jax.random.PRNGKey(0))
        act = jax.jit(algo.make_act(state, deterministic=deterministic))
        actions.append(act(observation, jax.random.PRNGKey(1)))
    np.testing.assert_array_equal(actions[0], actions[1])


def test_upstream_actor_initializes_on_kstar_native_action_space() -> None:
    env = RealisticWrappers(make("kstar_worldmodel"), max_steps=2)
    algo = _make_algo(
        env,
        num_envs=1,
        agent_kwargs={"hidden_layer_sizes": (4,)},
    )
    state = algo.init_state(jax.random.PRNGKey(0))
    action = jax.jit(algo.make_act(state, deterministic=True))(
        state.last_obs[0], jax.random.PRNGKey(1)
    )
    assert type(algo.actor) is GaussianPolicy
    assert action.shape == (6,)
    assert jnp.all(jnp.isfinite(action))
    assert jnp.all(action >= env.action_space.low)
    assert jnp.all(action <= env.action_space.high)


@pytest.mark.parametrize("quantized", [False, True], ids=("continuous", "quantized"))
def test_short_training_on_cheap_envelope_env_has_finite_outputs(quantized):
    env = CheapBoundaryEnv(
        obs_dim=2,
        action_low=(-1.0,),
        action_high=(1.0,),
    )
    if quantized:
        env = QuantizeActionWrapper(env=env, bin_counts=(3,))
    env = TruncationWrapper(env=env, max_steps=2)

    algo = _make_algo(
        env,
        total_timesteps=16,
        num_envs=2,
        num_steps=4,
        num_epochs=1,
        num_minibatches=1,
        eval_freq=16,
        normalize_rewards=False,
        normalize_observations=False,
    )
    _, (lengths, returns) = jax.jit(algo.train)(jax.random.PRNGKey(0))

    assert jnp.all(jnp.isfinite(returns))
    assert jnp.all(lengths > 0)


@pytest.mark.integration
class PPOTrainSmokeTest:
    def test_short_torax_training_run_completes_with_finite_outputs(self):
        env = RealisticWrappers(make("mock/circular/smoke", "mock"))
        algo = _make_algo(
            env,
            total_timesteps=1024,
            num_envs=16,
            num_steps=8,
            num_epochs=1,
            num_minibatches=2,
            eval_freq=1024,
            learning_rate=3e-4,
            normalize_rewards=False,
            normalize_observations=False,
        )
        _, (lengths, returns) = jax.jit(algo.train)(jax.random.PRNGKey(0))
        assert jnp.all(jnp.isfinite(returns))
        assert jnp.all(lengths > 0)
