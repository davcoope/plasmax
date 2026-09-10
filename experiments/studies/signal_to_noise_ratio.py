"""Signal-to-noise diagnostic for observation-noise multiplier selection.

Rolls out 8 episodes of iter/hybrid/flattop on bohm_gyrobohm with
zero observation noise. For each observed sensor, it compares the 
wrappers.yaml noise magnitude (at multiplier=1.0) against that sensor's 
natural std across episodes. Prints a per-sensor table including the 
noise/signal  ratio (noise std / observation std). 

Actions are held at zero each step by default. --random generates a 
random action within the actuator bounds for every step instead.
Zero and random actions bracket the true natural variation from opposite
sides - zero likely understates it (actuator-driven signals barely move 
on their own) and random likely overstates it (a trained policy explores a 
much narrower range than the full action space).

PhysicsRandomizationWrapper is included but not other wrappers - no 
need since no policy is being trained.
"""

import jax
import jax.numpy as jnp

from functools import partial

from plasmax.environment.factory import make
from plasmax.wrappers import PhysicsRandomizationWrapper

import tyro

env = make("iter/hybrid/flattop", backend="bohm_gyrobohm")
# Add physics randomisation wrapper.
# No need for other wrappers as no policy is being trained.
if env.plasmax_config.physics_randomization:
    env = PhysicsRandomizationWrapper(env)
layout = env.obs_layout()
# All observation names.
sensors = layout.profile_names + layout.scalar_names
# Per-sensor relative noise std, sourced from wrappers.yaml.
noise_cfg = env.plasmax_config.observations.realistic.noise


@partial(jax.jit, static_argnames=("n_steps", "random"))
def rollout(key, n_steps, random):
    key, action_key = jax.random.split(key)
    state, info = env.init(key)
    zero_action = jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    def body(carry, _):
        state, _, action_key = carry
        # random is a static arg, so this branches at trace time -
        # only one path is ever actually compiled.
        if random:
            action_key, sample_key = jax.random.split(action_key)
            action = jax.random.uniform(
                sample_key,
                env.action_space.shape,
                minval=env.action_space.low,
                maxval=env.action_space.high,
                dtype=env.action_space.dtype,
            )
        else:
            action = zero_action
        state, info = env.step(state, action)
        done = info.terminated | info.truncated
        return (state, info, action_key), (info.obs, done)

    (_, _, _), (obs_trace, done_trace) = jax.lax.scan(
        body, (state, info, action_key), None, n_steps
    )
    # Prepend the pre-step observation so the trace includes the start.
    obs_trace = jnp.concatenate([info.obs[None], obs_trace], axis=0)
    done_trace = jnp.concatenate([jnp.array([False]), done_trace], axis=0)
    # True from the step after the episode first ends onward, so a
    # disrupted/truncated episode's post-episode steps (which keep
    # running under scan with no auto-reset) can be excluded below.
    done_so_far = jnp.concatenate(
        [jnp.array([False]), jnp.cumsum(done_trace)[:-1] > 0]
    )
    valid = ~done_so_far
    return obs_trace, valid


# iter/hybrid/flattop episode length: 440s / 0.1s per step.
n_steps = 4400

def main(
        n_episodes: int = 64,
        random: bool = False
        )-> None:
    # Multiple independent episodes to improve estimate of natural_std.
    keys = jax.random.split(jax.random.key(0), n_episodes)
    obs_traces, valid_traces = jax.vmap(
        partial(rollout, n_steps=n_steps, random=random)
    )(keys)
    obs_flat = obs_traces.reshape(-1, obs_traces.shape[-1])
    valid_flat = valid_traces.reshape(-1)

    # Assess episode length
    completion_rate = float(jnp.mean(valid_traces[:, -1]))
    mean_valid_steps = float(jnp.sum(valid_flat)) / n_episodes
    print(
        f"completion_rate={completion_rate:.3f}  "
        f"mean_valid_steps={mean_valid_steps:.1f}"
    )

    header = (
        f"{'sensor':<18}{'rel_std':>8}{'obs_mean':>12}"
        f"{'natural_std':>14}{'noise@1.0x':>12}{'ratio':>10}"
    )
    print(header)
    for name in sensors:
        # Column indices this sensor occupies in the 24-length obs vector.
        sl = layout.slice_of(name)
        vals = obs_flat[:, sl]
        # Relative noise std for this sensor, from wrappers.yaml.
        rel_std = noise_cfg.get(name, 0.0)
        # Steps after an episode ends are set to NaN and ignored below.
        masked = jnp.where(valid_flat[:, None], vals, jnp.nan)
        obs_mean = float(jnp.nanmean(jnp.abs(masked)))
        # Natural spread across episodes — not noise from this script,
        # since no NoiseWrapper is applied here.
        natural_std = float(jnp.nanstd(masked))    
        # Noise std the wrapper would add at multiplier 1.0 (hypothetical
        # — reconstructed here, not actually applied).
        noise_1x = rel_std * obs_mean
        # Inverse signal-to-noise ratio: noise magnitude / natural signal
        # variation. >1 means noise dominates natural variation.
        ratio = noise_1x / natural_std if natural_std > 0 else float("nan")
        row = (
            f"{name:<18}{rel_std:>8.3f}{obs_mean:>12.4g}"
            f"{natural_std:>14.4g}{noise_1x:>12.4g}{ratio:>10.3f}"
        )
        print(row)

if __name__ == "__main__":
    tyro.cli(main)