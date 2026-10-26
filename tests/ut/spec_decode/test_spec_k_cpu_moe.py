# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for budgets and the hybrid executor; no A5 required."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.expert_offload.hybrid_executor import HybridExpertExecutor
from vllm_ascend.spec_k import SpecKBudgetState, SpecKPolicy, prune_cpu_moe_routes
from vllm_ascend.spec_k_config import SpecKConfig


class CpuBuffer:
    def __init__(self, *shape, dtype):
        self.cpu = torch.zeros(shape, dtype=dtype)
        self.np = self.cpu.numpy()


def state(capacity=2):
    policy = SpecKPolicy(SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128)), 6, torch.device("cpu"))
    return SpecKBudgetState(policy, capacity, 64, 32, CpuBuffer, contiguous_positions=True)


@pytest.mark.parametrize("apply_bonus", [False, True])
def test_post_markov_entropy_and_invalid_fallback(apply_bonus):
    policy = SpecKPolicy(
        SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128), apply_last_token=apply_bonus), 6, torch.device("cpu")
    )
    logits = torch.zeros(3, 7, 1024)
    logits[0, :, 0] = 30
    logits[2] = float("nan")
    budgets = policy.top_ks_from_logits(logits)
    assert budgets[0].tolist() == [3] * 7 + [3 if apply_bonus else 6]
    assert budgets[1].tolist() == [6] * 8
    assert budgets[2].tolist() == [6] * 8


def test_drafts_survive_unscheduling_growth_reordering_and_recompute():
    s = state(capacity=2)
    s.set_drafts(["a"], torch.tensor([[3, 4, 6]], dtype=torch.int32))
    s.set_drafts(["b"], torch.tensor([[4, 5, 6]], dtype=torch.int32))
    s.set_drafts(["c"], torch.tensor([[5, 6]], dtype=torch.int32))
    result = s.prepare(
        ["b", "a"], np.array([3, 3]), {"a": [1, 2], "b": [3, 4]}, [5, 5], torch.tensor([5, 6, 7, 5, 6, 7]), 8
    )
    assert result.tolist() == [4, 5, 6, 3, 4, 6, 6, 6]
    assert s.prepare(["c"], np.array([2]), {"c": [9]}, [5], torch.tensor([5, 6]), 2).tolist() == [5, 6]
    assert s.prepare(["b"], np.array([2]), {"b": [9]}, [5], torch.tensor([5, 6]), 2).tolist() == [4, 6]
    s.invalidate_drafts(["a"])
    assert s.prepare(["a"], np.array([3]), {}, [5], torch.tensor([5, 6, 7]), 4).tolist() == [3, 4, 6, 6]
    s.release(["a"])
    assert s.prepare(["d"], np.array([3]), {}, [5], torch.tensor([5, 6, 7]), 3).tolist() == [6, 6, 6]


def test_prefill_is_full_budget_when_pending_drafts_overlap_prompt():
    s = state()
    s.set_drafts(["a"], torch.tensor([[3, 3, 6]], dtype=torch.int32))
    assert s.prepare(["a"], np.array([3]), {"a": [1, 2]}, [8], torch.tensor([6, 7, 8]), 3).tolist() == [6, 6, 6]


def test_pruned_routes_skip_both_devices_and_weight_migration():
    calls = []

    class Backend:
        max_num_tokens = 8

        def submit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(wait=lambda: torch.zeros_like(kwargs["hidden_states"]))

    class Transfer:
        device_owned = True
        trace_transfer_count = 0

        def prepare(self, ids, mapping):
            self.ids = ids.clone()
            return SimpleNamespace(graph=False, wait_for_weights=lambda task: None)

    ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
    weights = torch.tensor([[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]])
    prune_cpu_moe_routes(weights, ids, torch.tensor([3], dtype=torch.int32))
    assert ids.tolist() == [[0, 1, 2, -1, -1, -1]]
    torch.testing.assert_close(weights, torch.tensor([[0.3, 0.25, 0.2, 0, 0, 0]]))
    transfer = Transfer()
    plan = HybridExpertExecutor(Backend(), 0).prepare(
        hidden_states=torch.ones(1, 4),
        topk_ids=ids,
        topk_weights=weights,
        log2phy=torch.tensor([0, -1, 1, -1, 2, -1]),
        weight_transfer=transfer,
    )
    assert transfer.ids.tolist() == [[0, 1, 2, -1, -1, -1]]
    assert calls[0]["topk_ids"].tolist() == [[-1, 1, -1, -1, -1, -1]]
    torch.testing.assert_close(plan.npu_topk_weights, torch.tensor([[0.3, 0, 0.2, 0, 0, 0]]))


@pytest.mark.parametrize(
    "field,value", [("use_v2_model_runner", True), ("adaptive", True), ("tp", 2), ("cpu_moe", False)]
)
def test_unsupported_combinations_fail_before_serving(field, value):
    cfg = SimpleNamespace(
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=SimpleNamespace(method="dspark", enable_adaptive_verification=False),
        lora_config=None,
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_experts_per_tok=6)),
    )
    ascend = SimpleNamespace(
        expert_offload_config=SimpleNamespace(
            expert_offload=True,
            cpu_moe=SimpleNamespace(enabled=True),
            enable_multi_card=False,
            experts_pruning_enabled=False,
            expert_substitution_enabled=False,
        ),
        dynamic_spec_config=SimpleNamespace(method=None),
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        enable_force_eplb=False,
        enable_reduce_sample=False,
        mix_placement=False,
        ascend_compilation_config=SimpleNamespace(enable_static_kernel=False),
    )
    if field == "adaptive":
        cfg.speculative_config.enable_adaptive_verification = value
    elif field == "tp":
        cfg.parallel_config.tensor_parallel_size = value
    elif field == "cpu_moe":
        ascend.expert_offload_config.cpu_moe.enabled = value
    else:
        cfg.use_v2_model_runner = value
    with pytest.raises(ValueError):
        SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128)).validate_runtime(cfg, ascend)


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="Requires Triton CPU interpreter")
@pytest.mark.parametrize("apply_bonus", [False, True])
def test_real_triton_entropy_metadata_and_negative_route_sentinel(apply_bonus):
    from vllm_ascend.ops.triton.spec_decode.spec_k import entropy_top_ks, prepare_budgets, prune_routes

    policy = SpecKPolicy(
        SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128), apply_last_token=apply_bonus), 6, torch.device("cpu")
    )
    logits = torch.zeros(2, 7, 8193)
    logits[0, :, 0] = 30
    expected = policy.top_ks_from_logits(logits)
    assert torch.equal(entropy_top_ks(logits, policy.log_thresholds, 6, apply_bonus), expected)
    history = torch.full((2, 512), 6, dtype=torch.int32)
    metadata = torch.tensor([[0, 1, 8, 0, 8, 4, 8], [1, 0, 8, 8, 8, 8, 8]], dtype=torch.int32)
    positions = torch.cat((torch.arange(4, 12), torch.arange(4, 12)))
    output = torch.full((24,), 6, dtype=torch.int32)
    histogram = torch.zeros(7, dtype=torch.int32)
    prepare_budgets(history, output, expected, metadata, positions, 512, 8, 6, histogram, contiguous_positions=True)
    ref = torch.cat((expected[1], torch.tensor([6] * 4), expected[0, 4:], torch.tensor([6] * 8)))
    assert torch.equal(output, ref)
    ids = torch.arange(6, dtype=torch.int32).repeat(24, 1)
    weights = torch.ones(24, 6)
    prune_routes(weights, ids, output, -1)
    keep = torch.arange(6)[None, :] < ref[:, None]
    assert torch.equal(ids, torch.where(keep, torch.arange(6)[None, :], -1))
    assert torch.equal(weights, keep.float())
    assert histogram.sum().item() == 12
    short_metadata = torch.tensor([[0, 0, 2, 0, 2, 0, 8]], dtype=torch.int32)
    prepare_budgets(history, output, expected, short_metadata, positions, 512, 2, 6, contiguous_positions=True)
    assert output[:2].tolist() == [3, 3 if apply_bonus else 6]
