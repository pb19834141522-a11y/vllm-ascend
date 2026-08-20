# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

def test_worker_patches_fused_moe_before_importing_models():
    import vllm_ascend.patch.worker  # noqa: F401
    from vllm.model_executor.models import qwen3_moe
    from vllm_ascend.patch.platform import patch_fused_moe

    assert qwen3_moe.FusedMoEFactory is patch_fused_moe._ascend_FusedMoE
