# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import routed_experts as routed_experts_module
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, EplbExpertTensorList


def _routed_experts(weight_views):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace(
        get_eplb_weight_views=lambda layer: weight_views,
    )
    return routed_experts


def test_get_expert_weights_flattens_layout_aware_views():
    weights = [torch.randn(2, 3, 4), torch.randn(2, 5)]

    views = list(_routed_experts(weights).get_expert_weights())

    assert [view.shape for view in views] == [torch.Size([2, 12]), torch.Size([2, 5])]
    assert views[0].untyped_storage().data_ptr() == weights[0].untyped_storage().data_ptr()


def test_get_expert_weights_preserves_independent_expert_tensors():
    expert_tensors = [torch.randn(3, 4), torch.randn(3, 4)]

    views = list(_routed_experts([expert_tensors]).get_expert_weights())

    assert len(views) == 1
    assert isinstance(views[0], EplbExpertTensorList)
    assert views[0].shape == torch.Size([2, 3, 4])
    assert all(actual is expected for actual, expected in zip(views[0], expert_tensors))

    buffer = torch.empty_like(views[0])
    assert isinstance(buffer, EplbExpertTensorList)
    assert buffer.shape == views[0].shape
    assert all(tensor.storage_offset() == 0 for tensor in buffer)


def test_get_expert_weights_rejects_unsupported_quantization():
    with pytest.raises(NotImplementedError, match="weight views are not defined"):
        list(_routed_experts([]).get_expert_weights())


def test_get_expert_weights_rejects_missing_weight_view_contract():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="must implement get_eplb_weight_views"):
        list(routed_experts.get_expert_weights())


def test_get_expert_weights_rejects_non_expert_first_dimension():
    with pytest.raises(ValueError, match="first dimension"):
        list(_routed_experts([torch.randn(3, 4)]).get_expert_weights())


def test_get_expert_weights_rejects_wrong_expert_tensor_list_length():
    with pytest.raises(ValueError, match="must contain local_num_experts"):
        list(_routed_experts([[torch.randn(3, 4)]]).get_expert_weights())


def test_get_expert_weights_rejects_non_contiguous_view():
    with pytest.raises(ValueError, match="flattenable without a copy"):
        list(_routed_experts([torch.randn(2, 3, 4).transpose(1, 2)]).get_expert_weights())


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_ascend_expert_map_follows_model_runner(use_v2_model_runner):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    upstream_map = torch.tensor([0, 1], dtype=torch.int32)
    object.__setattr__(routed_experts, "_use_v2_model_runner", use_v2_model_runner)
    routed_experts.ascend_expert_map = legacy_map
    object.__setattr__(routed_experts, "_expert_map", upstream_map)
    object.__setattr__(routed_experts, "rocm_aiter_fmoe_enabled", False)

    expected = upstream_map if use_v2_model_runner else legacy_map
    assert routed_experts.ascend_expert_map is expected


def test_update_expert_map_preserves_upstream_and_legacy_contracts(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    parent_update_calls = []

    def parent_update(instance):
        parent_update_calls.append(instance)

    monkeypatch.setattr(type(routed_experts).__mro__[1], "update_expert_map", parent_update)
    expert_map_manager = SimpleNamespace(_expert_map=None)
    object.__setattr__(routed_experts, "expert_map_manager", expert_map_manager)

    routed_experts.update_expert_map()

    assert parent_update_calls == [routed_experts]

    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    routed_experts.update_expert_map(legacy_map)

    assert routed_experts.ascend_expert_map is legacy_map
    assert expert_map_manager._expert_map is legacy_map


def test_mc2_dp_chunk_slices_spec_k_and_keeps_collective_count(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    object.__setattr__(routed_experts, "moe_config", SimpleNamespace(dp_rank=0, tp_size=1))
    object.__setattr__(routed_experts, "return_with_event", False)

    dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([384, 160]),
        local_sizes=None,
    )
    forward_context = SimpleNamespace(dp_metadata=dp_metadata)
    token_top_ks = torch.arange(384)
    extra_context = SimpleNamespace(
        max_tokens_across_dp=384,
        padded_num_tokens=384,
        mc2_mask=torch.ones(384, dtype=torch.bool),
        token_top_ks=token_top_ks,
        moe_comm_type=MoECommType.MC2,
    )
    monkeypatch.setattr(routed_experts_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(routed_experts_module, "get_mc2_tokens_capacity", lambda: 256)
    monkeypatch.setattr(routed_experts_module, "_EXTRA_CTX", extra_context)

    calls = []

    def fake_forward_impl(
        _self,
        *,
        hidden_states,
        router_logits,
        input_ids,
        _disable_chunking,
    ):
        calls.append(
            {
                "hidden": hidden_states.clone(),
                "router": router_logits.clone(),
                "input_ids": input_ids.clone(),
                "token_top_ks": extra_context.token_top_ks.clone(),
                "chunk_sizes": list(dp_metadata.local_sizes),
                "max_tokens": extra_context.max_tokens_across_dp,
                "padded_tokens": extra_context.padded_num_tokens,
                "mask": extra_context.mc2_mask.clone(),
                "disabled": _disable_chunking,
            }
        )
        return hidden_states + 1000

    object.__setattr__(
        routed_experts,
        "forward_impl",
        MethodType(fake_forward_impl, routed_experts),
    )
    hidden_states = torch.arange(384, dtype=torch.float32).unsqueeze(-1)
    router_logits = hidden_states + 10
    input_ids = torch.arange(384)

    output = AscendRoutedExperts._forward_impl_chunked(
        routed_experts,
        hidden_states=hidden_states,
        router_logits=router_logits,
        input_ids=input_ids,
        token_top_ks=token_top_ks,
    )

    torch.testing.assert_close(output, hidden_states + 1000)
    assert [call["chunk_sizes"] for call in calls] == [[256, 160], [128, 0]]
    assert [call["max_tokens"] for call in calls] == [256, 128]
    assert [call["padded_tokens"] for call in calls] == [256, 128]
    assert [int(call["mask"].sum()) for call in calls] == [256, 128]
    assert torch.equal(calls[0]["token_top_ks"], token_top_ks[:256])
    assert torch.equal(calls[1]["token_top_ks"], token_top_ks[256:])
    assert all(call["disabled"] for call in calls)
    assert dp_metadata.local_sizes is None
    assert extra_context.max_tokens_across_dp == 384
    assert extra_context.padded_num_tokens == 384
    assert extra_context.token_top_ks is token_top_ks


def test_mc2_dp_chunk_short_rank_uses_masked_dummy_chunk(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    object.__setattr__(routed_experts, "moe_config", SimpleNamespace(dp_rank=1, tp_size=1))
    object.__setattr__(routed_experts, "return_with_event", False)

    dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([384, 160]),
        local_sizes=None,
    )
    forward_context = SimpleNamespace(dp_metadata=dp_metadata)
    token_top_ks = torch.arange(160)
    extra_context = SimpleNamespace(
        max_tokens_across_dp=384,
        padded_num_tokens=384,
        mc2_mask=torch.ones(384, dtype=torch.bool),
        token_top_ks=token_top_ks,
        moe_comm_type=MoECommType.MC2,
    )
    monkeypatch.setattr(routed_experts_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(routed_experts_module, "get_mc2_tokens_capacity", lambda: 256)
    monkeypatch.setattr(routed_experts_module, "_EXTRA_CTX", extra_context)

    calls = []

    def fake_forward_impl(
        _self,
        *,
        hidden_states,
        router_logits,
        input_ids,
        _disable_chunking,
    ):
        calls.append(
            (
                hidden_states.clone(),
                extra_context.token_top_ks.clone(),
                extra_context.mc2_mask.clone(),
                list(dp_metadata.local_sizes),
            )
        )
        return hidden_states + 1000

    object.__setattr__(
        routed_experts,
        "forward_impl",
        MethodType(fake_forward_impl, routed_experts),
    )
    hidden_states = torch.arange(160, dtype=torch.float32).unsqueeze(-1)

    output = AscendRoutedExperts._forward_impl_chunked(
        routed_experts,
        hidden_states=hidden_states,
        router_logits=hidden_states + 10,
        input_ids=torch.arange(160),
        token_top_ks=token_top_ks,
    )

    torch.testing.assert_close(output, hidden_states + 1000)
    assert len(calls) == 2
    assert calls[0][0].shape[0] == 160
    assert calls[1][0].shape[0] == 1
    assert torch.equal(calls[1][1], token_top_ks[-1:])
    assert int(calls[1][2].sum()) == 0
    assert calls[1][3] == [128, 0]
