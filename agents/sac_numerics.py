"""Opt-in stage diagnostics around the upstream SAC implementation."""

from __future__ import annotations

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def check_numerics(stage: str, state: Any, **batches: Any) -> None:
    """Report and reject the first non-finite SAC stage without modifying state."""
    groups = {
        **batches,
        "actor_params": state.actor_ts.params,
        "actor_optimizer": state.actor_ts.opt_state,
        "critic_params": state.critic_ts.params,
        "critic_optimizer": state.critic_ts.opt_state,
        "target_params": state.critic_target_params,
        "alpha_params": state.alpha_ts.params,
        "alpha_optimizer": state.alpha_ts.opt_state,
        "obs_rms": state.obs_rms_state,
        "reward_rms": state.rew_rms_state,
        "alpha": jnp.exp(state.alpha_ts.params["log_alpha"]),
    }
    metrics = {}
    for name, group in groups.items():
        leaves = jax.tree.leaves(group)
        metrics[f"{name}/finite"] = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])
        )
        metrics[f"{name}/absmax"] = jnp.max(
            jnp.stack([jnp.max(jnp.abs(value)) for value in leaves])
        )
    for name, rms in (
        ("obs_rms", state.obs_rms_state),
        ("reward_rms", state.rew_rms_state),
    ):
        metrics[f"{name}/var_min"] = jnp.min(rms.var)
        metrics[f"{name}/var_max"] = jnp.max(rms.var)
    metrics["log_alpha"] = state.alpha_ts.params["log_alpha"]
    finite = jnp.all(
        jnp.stack([v for k, v in metrics.items() if k.endswith("/finite")])
    )

    def report(is_finite: Any, step: Any, values: dict[str, Any]) -> None:
        # A surrounding vmap can execute both branches of the traced cond.
        if bool(is_finite):
            return
        payload = {"stage": stage, "global_step": int(step)}
        for name, value in values.items():
            scalar = np.asarray(value).item()
            payload[name] = scalar if np.isfinite(scalar) else str(scalar)
        message = "SAC_NUMERICS " + json.dumps(payload, sort_keys=True, allow_nan=False)
        print(message, flush=True)
        raise FloatingPointError(message)

    def failed() -> None:
        jax.debug.callback(report, finite, state.global_step, metrics, ordered=True)

    jax.lax.cond(finite, lambda: None, failed)


__all__ = ["check_numerics"]
