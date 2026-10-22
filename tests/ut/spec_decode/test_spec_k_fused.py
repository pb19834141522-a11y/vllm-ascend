# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.spec_k import SpecKBudgetState, SpecKPolicy, prune_fused_routes, shard_token_budgets
from vllm_ascend.spec_k_config import SpecKConfig


@pytest.fixture(autouse=True)
def forbid_npu_initialization(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("This test must not initialize NPU")

    monkeypatch.setattr(torch.npu, "_lazy_init", fail)


class CpuBuffer:
    def __init__(self, *shape, dtype):
        self.cpu = torch.empty(shape, dtype=dtype)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, rows):
        return self.cpu[:rows]


def make_policy(thresholds=(128, 128, 128), apply_last_token=False):
    config = SpecKConfig(enabled=True, ppl_thresholds=thresholds, apply_last_token=apply_last_token)
    return SpecKPolicy(config, 6, torch.device("cpu"))


def make_state(max_reqs=3):
    return SpecKBudgetState(make_policy(), max_reqs, 64, 128, CpuBuffer)


def test_original_perplexity_policy_and_bonus_scope():
    logits = torch.full((3, 5, 8), -torch.inf)
    logits[0, :, :1] = 0
    logits[1, :, :2] = 0
    logits[2, :, :4] = 0
    result = make_policy((3.1469, 1.8222)).top_ks_from_logits(logits)
    assert result.tolist() == [[4] * 5 + [6], [5] * 5 + [6], [6] * 6]
    top3 = make_policy().top_ks_from_logits(logits)
    assert top3.tolist() == [[3] * 5 + [6]] * 3
    assert top3.float().mean().item() == 3.5
    assert make_policy(apply_last_token=True).top_ks_from_logits(logits).tolist() == [[3] * 6] * 3


def test_full_top6_control_and_invalid_distributions():
    logits = torch.randn(3, 5, 512)
    logits[0, 0] = -torch.inf
    logits[1, 2, 0] = torch.nan
    logits[2, 3, 0] = torch.inf
    assert torch.all(make_policy((1.0,)).top_ks_from_logits(logits) == 6)
    budgets = make_policy().top_ks_from_logits(logits)
    assert budgets[0, 0] == budgets[1, 2] == budgets[2, 3] == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "false"},
        {"apply_last_token": 1},
        {"ppl_thresholds": [True]},
        {"ppl_thresholds": [0]},
        {"ppl_thresholds": [float("nan")]},
        {"ppl_thresholds": [1, 2]},
        {"full_top_k_layer_range": [0, 4, 0]},
        {"full_top_k_layer_range": [False, 4]},
        {"unknown": True},
    ],
)
def test_bad_configs_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SpecKConfig(**kwargs)


def test_threshold_count_must_leave_an_active_expert():
    with pytest.raises(ValueError, match="fewer thresholds"):
        make_policy((1,) * 6)


def test_full_top_k_layer_slice():
    config = SpecKConfig(full_top_k_layer_range=(-3, 40))
    assert [i for i in range(40) if config.preserves_layer(i, 40)] == [37, 38, 39]
    assert not SpecKConfig().preserves_layer(0, 40)


def test_reserved_bucket_reduces_real_ep_work_without_renormalizing():
    ids = torch.tensor([[0, 1, 3, 4, 6, 7], [1, 2, 3, 4, 5, 6], [0, 2, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5]])
    weights = torch.tensor([[0.3, 0.2, 0.18, 0.14, 0.1, 0.08]] * 4)
    original_ids, original_weights = ids.clone(), weights.clone()
    budgets = torch.tensor([6, 3, 4, 0], dtype=torch.int32)
    prune_fused_routes(weights, ids, budgets, 8)
    active = torch.arange(6)[None, :] < budgets[:, None]
    assert torch.equal(ids[active], original_ids[active])
    assert torch.equal(weights[active], original_weights[active])
    assert torch.all(ids[~active] == 8)
    assert torch.all(weights[~active] == 0)
    expert_counts = torch.bincount(ids.flatten(), minlength=9)
    assert expert_counts[:8].sum() == 13
    assert expert_counts[8] == 11
    # Two EP ranks with four resident experts each. Only real expert buckets
    # contribute tokens to either rank's communication or GMM workloads.
    assert expert_counts[:4].sum() + expert_counts[4:8].sum() == budgets.sum()
    assert expert_counts[0] == (original_ids[active] == 0).sum()


def test_top6_is_bit_exact_and_graph_replay_reads_new_budget_values(monkeypatch):
    # Dynamo otherwise queries the default accelerator's stream even for CPU
    # tensors. Exercise the CPU graph without initializing an occupied NPU.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    weights = torch.rand(4, 6)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]] * 4)
    graphs = []

    def backend(graph, _):
        graphs.append(graph)
        return graph.forward

    def route(w, i, k):
        prune_fused_routes(w, i, k, 8)
        return w, i

    compiled = torch.compile(route, backend=backend, fullgraph=True)
    all6 = compiled(weights.clone(), ids.clone(), torch.full((4,), 6, dtype=torch.int32))
    assert torch.equal(all6[0], weights) and torch.equal(all6[1], ids)
    top3 = compiled(weights.clone(), ids.clone(), torch.full((4,), 3, dtype=torch.int32))
    assert torch.all(top3[0][:, 3:] == 0) and torch.all(top3[1][:, 3:] == 8)
    assert len(graphs) == 1


def test_request_reorder_new_request_and_partial_verification():
    state = make_state()
    a = [3, 4, 5, 3, 4, 6]
    b = [5, 3, 4, 5, 3, 6]
    state.set_drafts(["a", "b"], torch.tensor([a, b], dtype=torch.int32))
    result = state.prepare(
        ["b", "a", "new"],
        np.array([6, 3, 1]),
        {"b": [1] * 5, "a": [2] * 2},
        [5, 3, 2],
        torch.tensor([5, 6, 7, 8, 9, 10, 3, 4, 5, 0]),
        16,
    )
    assert result.tolist() == b + a[:3] + [6] * 7
    assert torch.equal(state.history[state.slots["a"], 3:6], torch.tensor(a[:3]))


def test_rejection_overwrites_uncommitted_rows_and_recompute_preserves_used_budgets():
    state = make_state()
    state.set_drafts(["a"], torch.tensor([[3, 3, 3, 3, 3, 6]], dtype=torch.int32))
    state.prepare(["a"], np.array([6]), {"a": [1] * 5}, [3], torch.arange(3, 9), 8)
    new = [5, 4, 3, 4, 5, 6]
    state.set_drafts(["a"], torch.tensor([new], dtype=torch.int32))
    state.prepare(["a"], np.array([6]), {"a": [1] * 5}, [3], torch.arange(4, 10), 8)
    state.set_drafts([], torch.empty((0, 6), dtype=torch.int32))
    recomputed = state.prepare(["a"], np.array([4]), {}, [3], torch.arange(3, 7), 8)
    assert recomputed[:4].tolist() == [3, 5, 4, 3]


def test_preemption_history_growth_and_finished_request_isolation():
    state = make_state(max_reqs=1)
    state.set_drafts(["a"], torch.tensor([[3, 3, 3, 3, 3, 6]], dtype=torch.int32))
    state.prepare(["a"], np.array([6]), {"a": [1] * 5}, [3], torch.arange(3, 9), 8)
    state.prepare(["b"], np.array([1]), {}, [3], torch.tensor([0]), 8)
    assert state.history.shape[0] == 2
    assert state.history[state.slots["a"], 3].item() == 3
    state.release(["a"])
    state.prepare(["c"], np.array([1]), {}, [0], torch.tensor([3]), 8)
    assert state.input_top_ks[0].item() == 6
    assert "a" not in state.previous_rows


def test_prompt_rows_remain_full_and_tp_dp_padding_aligns():
    state = make_state()
    state.set_drafts(["a"], torch.tensor([[3] * 6], dtype=torch.int32))
    result = state.prepare(["a"], np.array([6]), {"a": [1] * 5}, [3], torch.arange(6), 8)
    assert result.tolist() == [6, 6, 6, 3, 3, 3, 6, 6]
    values = torch.arange(17, dtype=torch.int32)
    shards = [shard_token_budgets(values, 3, 8, rank) for rank in range(8)]
    assert torch.equal(torch.cat(shards), torch.cat((values, torch.zeros(7, dtype=torch.int32))))
    assert shard_token_budgets(torch.tensor([3, 4]), 4, 1, 0).tolist() == [3, 4, 0, 0]


def test_sequence_parallel_slices_before_dp_padding():
    budgets = torch.tensor([3, 4, 5, 6], dtype=torch.int32)
    assert shard_token_budgets(budgets, 4, 2, 0, sequence_parallel=True).tolist() == [3, 4, 0, 0]
    assert shard_token_budgets(budgets, 4, 2, 1, sequence_parallel=True).tolist() == [5, 6, 0, 0]
    assert shard_token_budgets(budgets, 4, 2, 0).tolist() == [3, 4, 5, 6]
    assert shard_token_budgets(budgets, 4, 2, 1).tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("drafts", [1, 5, 7])
def test_budget_width_follows_configured_draft_length(drafts):
    policy = make_policy()
    budgets = policy.top_ks_from_logits(torch.zeros(2, drafts, 8))
    assert budgets.shape == (2, drafts + 1)
    assert budgets[:, :-1].tolist() == [[3] * drafts] * 2
    assert budgets[:, -1].tolist() == [6, 6]
    state = make_state()
    state.set_drafts(["a", "b"], budgets)
    positions = torch.arange(4, 4 + drafts + 1).repeat(2)
    actual = state.prepare(
        ["b", "a"],
        np.array([drafts + 1] * 2),
        {"a": [0] * drafts, "b": [0] * drafts},
        [4, 4],
        positions,
        (drafts + 1) * 2,
    )
    assert actual.tolist() == ([3] * drafts + [6]) * 2


@pytest.mark.parametrize("mode", ["dynamic", "top6", "draft", "preserved_layer", "disabled", "wrong_path"])
def test_routed_experts_applies_budget_after_physical_id_mapping(monkeypatch, mode):
    import vllm_ascend.ops.fused_moe.routed_experts as module
    from vllm_ascend.ascend_forward_context import MoECommType

    ids = torch.tensor([[0, 1, 2, 3, 4, 5]] * 2)
    weights = torch.tensor([[0.3, 0.2, 0.18, 0.14, 0.1, 0.08]] * 2)
    layer = SimpleNamespace(
        router=SimpleNamespace(_select_experts=lambda **kwargs: (weights, ids)),
        log2phy=torch.tensor([7, 6, 5, 4, 3, 2, 1, 0]),
        n_shared_experts=0,
        mix_placement=False,
        global_redundant_expert_num=0,
        moe_config=SimpleNamespace(num_experts=8, is_sequence_parallel=False),
        _spec_k_enabled=mode != "disabled",
        _spec_k_full_top_k=mode == "preserved_layer",
    )
    context = SimpleNamespace(
        is_draft_model=mode == "draft",
        token_top_ks=torch.tensor([6, 6] if mode == "top6" else [3, 4]),
        moe_comm_type=MoECommType.MC2 if mode == "wrong_path" else MoECommType.FUSED_MC2,
        use_mega_moe=False,
    )
    monkeypatch.setattr(module, "_EXTRA_CTX", context)
    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(enable_force_eplb=False))
    monkeypatch.setattr(module, "get_moe_num_logical_experts", lambda *args, **kwargs: 8)
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    if mode == "wrong_path":
        with pytest.raises(RuntimeError, match="fused MC2 path"):
            module.AscendRoutedExperts._select_experts(layer, torch.empty(2, 8), torch.empty(2, 8), False)
        return
    result_w, result_ids = module.AscendRoutedExperts._select_experts(
        layer,
        torch.empty(2, 8),
        torch.empty(2, 8),
        False,
    )
    expected_ids = torch.tensor([[7, 6, 5, 4, 3, 2]] * 2)
    expected_w = torch.tensor([[0.3, 0.2, 0.18, 0.14, 0.1, 0.08]] * 2)
    if mode == "dynamic":
        expected_ids[0, 3:], expected_ids[1, 4:] = 8, 8
        expected_w[0, 3:], expected_w[1, 4:] = 0, 0
    assert torch.equal(result_ids, expected_ids)
    assert torch.equal(result_w, expected_w)


def test_runtime_config_requires_the_supported_fused_path(monkeypatch):
    from vllm_ascend import ascend_config
    from vllm_ascend.device.hardware_profile import MoECommPolicy

    monkeypatch.setattr(ascend_config, "is_mega_moe_supported", lambda: False)
    monkeypatch.setattr(
        ascend_config,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(
            moe_comm_policy=MoECommPolicy.FUSED_OR_CAPACITY,
        ),
    )
    cfg = SimpleNamespace(
        spec_k_config=SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128)),
        enable_fused_mc2=1,
        mix_placement=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        enable_force_eplb=False,
        dynamic_spec_config=SimpleNamespace(method=None),
    )
    vc = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark", enable_adaptive_verification=False),
        use_v2_model_runner=False,
        lora_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            world_size_across_dp=8,
        ),
        model_config=SimpleNamespace(quantization="ascend", hf_text_config=SimpleNamespace(num_experts_per_tok=6)),
    )
    ascend_config.AscendConfig._validate_spec_k(cfg, vc)
    cfg.enable_fused_mc2 = 0
    with pytest.raises(ValueError, match="fused MC2"):
        ascend_config.AscendConfig._validate_spec_k(cfg, vc)
    cfg.enable_fused_mc2 = 1
    vc.use_v2_model_runner = True
    ascend_config.AscendConfig._validate_spec_k(cfg, vc)
    vc.speculative_config.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(dspark_draft_topk=32))
    with pytest.raises(ValueError, match="full-vocabulary"):
        ascend_config.AscendConfig._validate_spec_k(cfg, vc)
