# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import (
    MoEPrepareOutput,
)
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoETokenDispatchInput,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.routed_experts import (
    AscendRoutedExperts,
    _apply_token_top_ks,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    TokenDispatcherWithAllGather,
)
from vllm_ascend.quantization.quant_type import QuantType


def test_apply_token_top_ks_masks_routes_per_token():
    topk_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    topk_weights = torch.ones(2, 4)

    _apply_token_top_ks(
        topk_ids,
        topk_weights,
        invalid_expert_id=8,
        token_top_ks=torch.tensor([3, 2], dtype=torch.int32),
    )

    assert topk_ids.tolist() == [[0, 1, 2, 8], [4, 5, 8, 8]]
    assert topk_weights.tolist() == [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_apply_token_top_ks_rejects_misaligned_shape():
    with pytest.raises(ValueError, match="shape must match"):
        _apply_token_top_ks(
            torch.zeros(2, 4, dtype=torch.int32),
            torch.ones(2, 4),
            invalid_expert_id=4,
            token_top_ks=torch.ones(3, dtype=torch.int32),
        )


def test_allgather_w4a8_spec_k_uses_bf16_routing(monkeypatch):
    hidden_states = torch.randn(1, 4)
    topk_weights = torch.tensor([[0.75, 0.0]])
    # Four logical experts use IDs [0, 4). Spec-K uses 4 as the dropped-route
    # sentinel, so this also covers bounds-safe expert-map lookup.
    topk_ids = torch.tensor([[0, 4]], dtype=torch.int32)
    expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)

    dispatcher = TokenDispatcherWithAllGather(
        apply_router_weight_on_input=False,
        top_k=2,
        max_num_tokens=8,
        ep_size=2,
        num_experts=4,
        num_local_experts=2,
    )
    dispatch_input = MoETokenDispatchInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        routing=MoeRouterInput(
            expert_map=expert_map,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
            pertoken_scale=None,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W4A8),
    )

    init_routing = MagicMock(
        return_value=(
            torch.randn(2, 4),
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([1, 0], dtype=torch.int64),
            torch.ones(2),
        )
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.token_dispatcher.get_ascend_config",
        lambda: SimpleNamespace(spec_k_config=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.token_dispatcher.DeviceOperator.npu_moe_init_routing",
        init_routing,
    )

    output = dispatcher.token_dispatch(dispatch_input)

    assert init_routing.call_args.kwargs["quant_mode"] == -1
    assert output.dynamic_scale is None
    assert output.combine_metadata.topk_weights.tolist() == [[0.75, 0.0]]


@pytest.mark.parametrize("full_top_k", [False, True])
def test_routed_experts_applies_spec_k_after_prepare(monkeypatch, full_top_k):
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 8)
    token_top_ks = torch.tensor([2, 3], dtype=torch.int32)
    expected_top_ks = None if full_top_k else token_top_ks

    router = MagicMock()
    router._select_experts.return_value = (
        torch.ones(2, 4),
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32),
    )
    router.eplb_state = None
    quant_method = MagicMock()
    quant_method.quant_method = None
    quant_method.apply.return_value = FusedExpertsResult(hidden_states)
    moe_comm_method = MagicMock()
    moe_comm_method.prepare.return_value = MoEPrepareOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        mc2_mask=None,
        padded_hidden_states_shape=None,
        token_top_ks=expected_top_ks,
    )
    moe_comm_method.finalize.return_value = hidden_states

    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.router = router
    routed_experts.quant_method = quant_method
    routed_experts._spec_k_full_top_k = full_top_k
    routed_experts.enable_npugraph_ex_static_kernel = False
    routed_experts.moe_config = SimpleNamespace(num_experts=8)
    routed_experts.n_shared_experts = 0
    routed_experts.mix_placement = False
    routed_experts.log2phy = None
    routed_experts.global_redundant_expert_num = 0
    routed_experts.dynamic_eplb = False
    routed_experts._use_v2_model_runner = False
    routed_experts.return_with_event = False

    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.routed_experts.get_forward_context",
        lambda: SimpleNamespace(all_moe_layers=None, moe_layer_index=0),
    )
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.routed_experts._EXTRA_CTX",
        SimpleNamespace(
            token_top_ks=token_top_ks,
            in_profile_run=False,
            moe_comm_method=moe_comm_method,
            flash_comm_v1_enabled=False,
            eplb_heat_collection_status=False,
        ),
    )

    result = routed_experts.forward_impl(
        hidden_states=hidden_states,
        router_logits=router_logits,
        _disable_chunking=True,
    )

    assert result is hidden_states
    assert (
        moe_comm_method.prepare.call_args.kwargs["token_top_ks"]
        is expected_top_ks
    )
    apply_kwargs = quant_method.apply.call_args.kwargs
    if full_top_k:
        assert apply_kwargs["topk_ids"].tolist() == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    else:
        assert apply_kwargs["topk_ids"].tolist() == [
            [0, 1, 8, 8],
            [4, 5, 6, 8],
        ]
        assert apply_kwargs["topk_weights"].tolist() == [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]


def test_moe_prepare_output_preserves_pertoken_scale_position():
    pertoken_scale = torch.ones(2)
    output = MoEPrepareOutput(
        torch.ones(2, 4),
        torch.ones(2, 8),
        None,
        None,
        pertoken_scale,
    )

    assert output.pertoken_scale is pertoken_scale
    assert output.token_top_ks is None
