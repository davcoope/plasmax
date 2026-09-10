"""Signal-to-noise diagnostic for observation-noise multiplier selection.

Rolls out 8 episodes of iter/hybrid/flattop on bohm_gyrobohm with
zero observation noise. For each observed sensor, it compares the 
wrappers.yaml noise magnitude (at multiplier=1.0) against that sensor's 
natural std across episodes. Prints a per-sensor table including the 
noise/signal  ratio (noise std / observation std).
"""

import jax
import jax.numpy as jnp

from functools import partial

from plasmax.environment.factory import make

env = make("iter/hybrid/flattop", backend="bohm_gyrobohm")
layout = env.obs_layout()
# All observation names.
sensors = layout.profile_names + layout.scalar_names
# Per-sensor relative noise std, sourced from wrappers.yaml.
noise_cfg = env.plasmax_config.observations.realistic.noise


@partial(jax.jit, static_argnames=("n_steps",))
def rollout(key, n_steps):
    state, info = env.init(key)
    action = jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    def body(carry, _):
        state, _ = carry
        state, info = env.step(state, action)
        return (state, info), info.obs

    # obs_trace: (n_steps, 24) - one row per timestep.
    _, obs_trace = jax.lax.scan(body, (state, info), None, n_steps)
    # Prepend the pre-step observation so the trace includes the start.
    return jnp.concatenate([info.obs[None], obs_trace], axis=0)


# Multiple independent episodes to improve estimate of natural_std.
n_episodes = 8
# iter/hybrid/flattop episode length: 440s / 0.1s per step.
n_steps = 4400
keys = jax.random.split(jax.random.key(0), n_episodes)
obs_traces = jax.vmap(rollout, in_axes=(0, None))(keys, n_steps)
obs_flat = obs_traces.reshape(-1, obs_traces.shape[-1])

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
    obs_mean = float(jnp.mean(jnp.abs(vals)))
    # Natural spread across episodes — not noise from this script,
    # since no NoiseWrapper is applied here.
    natural_std = float(jnp.std(vals))
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