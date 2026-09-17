# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import experts_selector


def _enable_spec_k(monkeypatch, token_top_ks):
    monkeypatch.setattr(
        experts_selector,
        "get_ascend_config",
        lambda: SimpleNamespace(spec_k_config=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(
        experts_selector,
        "_EXTRA_CTX",
        SimpleNamespace(token_top_ks=token_top_ks),
    )


def test_apply_dynamic_top_k_masks_routes_per_token(monkeypatch):
    _enable_spec_k(
        monkeypatch,
        torch.tensor([3, 2], dtype=torch.int32),
    )
    topk_ids = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32
    )
    topk_weights = torch.ones(2, 4)

    experts_selector.apply_dynamic_top_k(
        topk_weights,
        topk_ids,
        SimpleNamespace(_spec_k_full_top_k=False),
    )

    assert topk_ids.tolist() == [[0, 1, 2, -1], [4, 5, -1, -1]]
    assert topk_weights.tolist() == [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_apply_dynamic_top_k_preserves_full_top_k_layer(monkeypatch):
    _enable_spec_k(monkeypatch, torch.tensor([1], dtype=torch.int32))
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    topk_weights = torch.ones(1, 3)

    experts_selector.apply_dynamic_top_k(
        topk_weights,
        topk_ids,
        SimpleNamespace(_spec_k_full_top_k=True),
    )

    assert topk_ids.tolist() == [[0, 1, 2]]
    assert topk_weights.tolist() == [[1.0, 1.0, 1.0]]


def test_apply_dynamic_top_k_rejects_misaligned_shape(monkeypatch):
    _enable_spec_k(monkeypatch, torch.ones(3, dtype=torch.int32))

    with pytest.raises(ValueError, match="shape must match"):
        experts_selector.apply_dynamic_top_k(
            torch.ones(2, 4),
            torch.zeros(2, 4, dtype=torch.int32),
            SimpleNamespace(_spec_k_full_top_k=False),
        )
