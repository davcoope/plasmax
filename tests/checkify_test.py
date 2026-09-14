"""Explicit reward checks compose with JAX batching and loop gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

import plasmax  # noqa: F401 -- install the package's narrow upstream fixes


def _loop_reward(value: jax.Array) -> jax.Array:
    result = jax.lax.while_loop(lambda x: x < 3.0, lambda x: x + 1.0, value)
    checkify.check(jnp.isfinite(result), "Reward must be finite")
    return result


@pytest.mark.parametrize("check_before_vmap", [False, True])
def test_batched_while_supports_both_checkify_orders(check_before_vmap: bool) -> None:
    if check_before_vmap:
        function = jax.vmap(
            checkify.checkify(_loop_reward, errors=checkify.user_checks)
        )
    else:
        function = checkify.checkify(
            jax.vmap(_loop_reward), errors=checkify.user_checks
        )
    compiled = jax.jit(function)
    error, rewards = compiled(jnp.array([0.0, 1.0]))
    error.throw()
    np.testing.assert_array_equal(rewards, [3.0, 3.0])

    error, _ = compiled(jnp.array([0.0, jnp.inf]))
    with pytest.raises(checkify.JaxRuntimeError, match="Reward must be finite"):
        error.throw()


def test_checks_inside_effectful_loops_are_preserved() -> None:
    def checked_loop(value: jax.Array) -> jax.Array:
        def body(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            index, current = carry
            checkify.check(current >= 0.0, "Negative loop value")
            return index + 1, current + 1.0

        return jax.lax.while_loop(
            lambda carry: carry[0] < 2, body, (jnp.int32(0), value)
        )[1]

    checked = jax.jit(
        jax.vmap(checkify.checkify(checked_loop, errors=checkify.user_checks))
    )
    error, values = checked(jnp.array([1.0, 2.0]))
    error.throw()
    np.testing.assert_array_equal(values, [3.0, 4.0])
    error, _ = checked(jnp.array([1.0, -1.0]))
    with pytest.raises(checkify.JaxRuntimeError, match="Negative loop value"):
        error.throw()


def test_nested_checked_batches_preserve_custom_loop_gradients() -> None:
    @jax.custom_jvp
    def dynamics(value: jax.Array) -> jax.Array:
        return jax.lax.while_loop(lambda x: x < 3.0, lambda x: x * 2.0, value)

    @dynamics.defjvp
    def dynamics_jvp(
        primals: tuple[jax.Array], tangents: tuple[jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        result = dynamics(primals[0])
        return result, tangents[0] * result / primals[0]

    def reward(value: jax.Array) -> jax.Array:
        result = dynamics(value)
        checkify.check(jnp.isfinite(result), "Reward must be finite")
        return result

    checked_reward = jax.jit(checkify.checkify(reward, errors=checkify.user_checks))

    def nested(value: jax.Array) -> jax.Array:
        error, result = checked_reward(value)
        checkify.check_error(error)
        return result

    checked_gradients = jax.jit(
        checkify.checkify(
            jax.vmap(jax.vmap(jax.grad(nested))), errors=checkify.user_checks
        )
    )
    error, gradients = checked_gradients(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    error.throw()
    np.testing.assert_allclose(gradients, [[4.0, 2.0], [1.0, 1.0]], rtol=0.0, atol=0.0)
