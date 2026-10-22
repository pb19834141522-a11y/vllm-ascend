# SPDX-License-Identifier: Apache-2.0

import math
from numbers import Real
from typing import Any

from pydantic import ConfigDict, field_validator, model_validator

from vllm_ascend.config_utils import config


@config(config=ConfigDict(frozen=True, extra="forbid"))
class SpecKConfig:
    """Dependency-light DSpark configuration, also used during platform discovery."""

    enabled: bool = False
    ppl_thresholds: tuple[float, ...] = ()
    full_top_k_layer_range: tuple[int, ...] = (0, 0)
    apply_last_token: bool = False
    log_stats_interval: int = 100

    @field_validator("log_stats_interval", mode="before")
    @classmethod
    def _stats_interval(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Spec-K log_stats_interval must be a nonnegative integer (0 disables logging)")
        return value

    @field_validator("enabled", "apply_last_token", mode="before")
    @classmethod
    def _strict_bool(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("Spec-K switches must be bools")
        return value

    @field_validator("ppl_thresholds", mode="before")
    @classmethod
    def _numeric_thresholds(cls, value: Any):
        if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, Real) for item in value
        ):
            raise ValueError("spec_k_config.ppl_thresholds must be a list of numbers")
        return value

    @field_validator("full_top_k_layer_range", mode="before")
    @classmethod
    def _layer_range(cls, value: Any):
        if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
            raise ValueError("full_top_k_layer_range must contain 2 or 3 integers")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("full_top_k_layer_range must contain integers")
        if len(value) == 3 and value[2] == 0:
            raise ValueError("full_top_k_layer_range step cannot be zero")
        return value

    @model_validator(mode="after")
    def _validate_thresholds(self):
        if self.enabled and not self.ppl_thresholds:
            raise ValueError("Spec-K requires nonempty ppl_thresholds")
        if any(not math.isfinite(value) or value <= 0 for value in self.ppl_thresholds):
            raise ValueError("Spec-K thresholds must be finite and positive")
        if any(left < right for left, right in zip(self.ppl_thresholds, self.ppl_thresholds[1:])):
            raise ValueError("Spec-K thresholds must be in non-increasing order")
        return self

    def preserves_layer(self, layer_index: int, num_layers: int) -> bool:
        return layer_index in range(*slice(*self.full_top_k_layer_range).indices(num_layers))
