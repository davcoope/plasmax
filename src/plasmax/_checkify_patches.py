"""Narrow JAX fixes for explicit checks around vmapped TORAX while loops.

Effect-free loops contain no user checks, so checkify can preserve them intact.
JAX's DCE sink is a zero-output primitive; its batching rule must keep that arity.
Remove these fixes when upstream JAX supports both contracts.
"""

from typing import Any

from jax._src import checkify as checkify_impl
from jax._src.interpreters import batching
from jax._src.lax import lax
from jax._src.lax.control_flow import loops
from jax.experimental import checkify

_original_while_check = checkify_impl.error_checks[loops.while_p]


def _check_user_while(
    error: checkify.Error,
    enabled_errors: Any,
    *args: Any,
    cond_jaxpr: Any,
    body_jaxpr: Any,
    **params: Any,
) -> Any:
    if (
        enabled_errors == checkify.user_checks
        and cond_jaxpr.out_avals[0].shape
        and not cond_jaxpr.effects
        and not body_jaxpr.effects
    ):
        return error, loops.while_p.bind(
            *args, cond_jaxpr=cond_jaxpr, body_jaxpr=body_jaxpr, **params
        )
    return _original_while_check(
        error,
        enabled_errors,
        *args,
        cond_jaxpr=cond_jaxpr,
        body_jaxpr=body_jaxpr,
        **params,
    )


def _batch_dce_sink(args: Any, _dims: Any, **params: Any) -> Any:
    return lax.dce_sink_p.bind(*args, **params), []


checkify_impl.error_checks[loops.while_p] = _check_user_while
batching.primitive_batchers[lax.dce_sink_p] = _batch_dce_sink
