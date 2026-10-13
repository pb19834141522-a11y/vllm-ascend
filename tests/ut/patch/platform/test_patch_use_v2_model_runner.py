# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform import patch_use_v2_model_runner  # noqa: F401


@pytest.mark.parametrize(
    ("method", "adaptive_verification", "pcp_size", "expected"),
    [
        ("dspark", False, 1, []),
        ("dspark", True, 1, ["adaptive draft verification"]),
        ("dspark", False, 2, ["prefill context parallel"]),
        ("dflash", False, 1, []),
    ],
)
def test_ascend_v1_allows_dspark_but_preserves_other_restrictions(
    method, adaptive_verification, pcp_size, expected
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            enable_batch_sharded_sampling=False,
        ),
        speculative_config=SimpleNamespace(
            method=method,
            enable_adaptive_verification=adaptive_verification,
        ),
        model_config=None,
        _dflash_needs_multi_kv_group=lambda: False,
        _is_dflash2_draft=lambda: False,
    )

    assert VllmConfig._get_v1_model_runner_unsupported_features(config) == expected
