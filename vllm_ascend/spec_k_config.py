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

    def validate_runtime(self, vllm_config, ascend_config) -> None:
        if not self.enabled:
            return
        parallel = vllm_config.parallel_config
        spec = vllm_config.speculative_config
        offload = ascend_config.expert_offload_config
        if vllm_config.use_v2_model_runner or spec is None or spec.method != "dspark":
            raise ValueError("CPU MoE Spec-K requires Model Runner V1 and DSpark")
        if (
            any(
                getattr(parallel, key, 1) != 1
                for key in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            )
            or offload.enable_multi_card
        ):
            raise ValueError("This CPU MoE Spec-K port supports TP1/PP1/DP1 without context parallelism")
        if getattr(spec, "disable_padded_drafter_batch", False):
            raise ValueError("CPU MoE Spec-K requires the default padded DSpark request batch")
        if not offload.expert_offload or not offload.cpu_moe.enabled:
            raise ValueError("CPU MoE Spec-K requires expert_offload and cpu_moe.enabled")
        if (
            getattr(spec, "enable_adaptive_verification", False)
            or getattr(spec, "num_speculative_tokens_per_batch_size", None)
            or ascend_config.dynamic_spec_config.method is not None
        ):
            raise ValueError("CPU MoE Spec-K requires fixed DSpark verification length")
        if (
            vllm_config.lora_config is not None
            or vllm_config.kv_transfer_config is not None
            or ascend_config.eplb_config.dynamic_eplb
            or ascend_config.enable_force_eplb
            or ascend_config.enable_reduce_sample
            or ascend_config.mix_placement
            or ascend_config.ascend_compilation_config.enable_static_kernel
        ):
            raise ValueError(
                "CPU MoE Spec-K is incompatible with LoRA, KV transfer, EPLB, "
                "reduced sampling, mixed placement or static kernels"
            )
        activation = getattr(ascend_config, "activation_routing", {})
        if activation is not None and activation.get("enabled", False):
            raise ValueError("Disable activation routing when enabling CPU MoE Spec-K")
        if offload.experts_pruning_enabled or offload.expert_substitution_enabled:
            raise ValueError("Disable independent expert pruning/substitution for the Spec-K control experiment")
        hf = vllm_config.model_config.hf_text_config
        top_k = getattr(hf, "num_experts_per_tok", getattr(hf, "top_k_experts", 0))
        if top_k <= len(self.ppl_thresholds):
            raise ValueError("Spec-K needs fewer thresholds than target experts per token")
