# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.patch.worker import patch_qwen3_dflash


def test_precompute_converts_context_positions_to_int64():
    rotary_emb = MagicMock()
    self_attn = SimpleNamespace(k_norm=torch.nn.Identity(), rotary_emb=rotary_emb)
    model = SimpleNamespace(
        _num_attn_layers=1,
        _kv_size=2,
        _head_dim=2,
        _num_kv_heads=1,
        hidden_norm=torch.nn.Identity(),
        _fused_kv_weight=torch.ones((4, 3)),
        _fused_kv_bias=torch.zeros(4),
        layers=[SimpleNamespace(self_attn=self_attn)],
    )

    patch_qwen3_dflash.precompute_and_store_context_kv(
        model,
        context_states=torch.ones((2, 3)),
        context_positions=torch.tensor([0, 1], dtype=torch.int32),
    )

    positions = rotary_emb.call_args.args[0]
    assert positions.dtype is torch.int64


def test_forward_converts_positions_to_int64():
    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([0, 1], dtype=torch.int32)
    input_embeds = torch.ones((2, 3))
    original_forward = MagicMock(return_value=torch.ones((2, 3)))

    with patch.object(
        patch_qwen3_dflash,
        "_orig_dflash_model_forward",
        original_forward,
    ):
        result = patch_qwen3_dflash._patched_dflash_model_forward(
            SimpleNamespace(), input_ids, positions, input_embeds=input_embeds
        )

    assert result is original_forward.return_value
    assert original_forward.call_args.args[2].dtype is torch.int64
    assert original_forward.call_args.args[3] is input_embeds


def test_forward_preserves_int64_positions():
    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    original_forward = MagicMock(return_value=torch.ones((2, 3)))

    with patch.object(
        patch_qwen3_dflash,
        "_orig_dflash_model_forward",
        original_forward,
    ):
        patch_qwen3_dflash._patched_dflash_model_forward(
            SimpleNamespace(), input_ids, positions
        )

    assert original_forward.call_args.args[2] is positions
