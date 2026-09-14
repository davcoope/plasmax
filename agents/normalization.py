"""Keep the Envelope float32 boundary around upstream Rejax normalization."""

from __future__ import annotations

from typing import Any

import jax
from rejax.algos.algorithm import register_init

from training.envelope_gymnax import EnvelopeGymnax


class EnvelopeNormalizationMixin:
    """Adapt normalization setup without replacing any upstream updates."""

    @classmethod
    def create(cls, **config: Any) -> Any:
        env = config.get("env")
        instance = super().create(**config)
        if isinstance(env, EnvelopeGymnax) and instance.normalize_observations:
            # Envelope already supplies float32 observations. Rejax's redundant
            # FloatObsWrapper instead casts them to float64 while TORAX uses x64.
            instance = instance.replace(env=env)
        return instance

    @register_init
    def initialize_reward_rms_state(self, rng: jax.Array) -> dict[str, Any]:
        if isinstance(self.env, EnvelopeGymnax):
            # Only the upstream normalizer initializer changes default dtype;
            # TORAX initialization and transitions retain their x64 context.
            with jax.enable_x64(False):
                return super().initialize_reward_rms_state(rng)
        return super().initialize_reward_rms_state(rng)


__all__ = ["EnvelopeNormalizationMixin"]
