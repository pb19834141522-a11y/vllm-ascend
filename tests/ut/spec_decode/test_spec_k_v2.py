# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.ascend_forward_context import get_mrv2_token_top_ks, override_mrv2_token_top_ks
from vllm_ascend.spec_k import SpecKBudgetState, SpecKDraftBudgetState, SpecKPolicy
from vllm_ascend.spec_k_config import SpecKConfig
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.fixture(autouse=True)
def forbid_npu(monkeypatch):
    monkeypatch.setattr(
        torch.npu, "_lazy_init", Mock(side_effect=AssertionError("CPU contract must not initialize NPU"))
    )


class CpuBuffer:
    def __init__(self, *shape, dtype):
        self.cpu = torch.zeros(shape, dtype=dtype)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, rows):
        return self.cpu[:rows].clone()


def policy(thresholds=(3.1469, 1.8222), apply_bonus=False):
    return SpecKPolicy(
        SpecKConfig(enabled=True, ppl_thresholds=thresholds, apply_last_token=apply_bonus), 6, torch.device("cpu")
    )


def budget_state():
    return SpecKBudgetState(policy(), 4, 64, 64, CpuBuffer)


@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("apply_bonus", [False, True])
def test_budget_output_is_persistent_and_request_indexed(steps, apply_bonus):
    p = policy(apply_bonus=apply_bonus)
    state = SpecKDraftBudgetState(p, 5, steps, 8)
    pointer = state.budgets.data_ptr()
    logits = torch.zeros(3, steps, 8)
    logits[0, :, 0] = 20
    logits[2, :, 0] = 30
    indices = torch.tensor([4, 1, -1]).repeat_interleave(steps)
    for step in range(steps):
        state.record(logits[:, step], step)
    state.finish(3, indices)
    expected = p.top_ks_from_logits(logits)
    assert torch.equal(state.budgets[4], expected[0])
    assert torch.equal(state.budgets[1], expected[1])
    assert torch.all(state.budgets[[0, 2, 3]] == 6)
    # A different active batch must retain the absent request's pending draft.
    for step in range(steps):
        state.record(torch.zeros(1, 8), step)
    state.finish(1, torch.tensor([2]).repeat_interleave(steps))
    assert torch.equal(state.budgets[4], expected[0])
    assert state.budgets.data_ptr() == pointer
    state.finish(0, torch.empty(0, dtype=torch.int32))


def test_sampling_hook_uses_corrected_logits_without_changing_upstream_drafts():
    steps, rows, vocab = 7, 3, 8
    sample_hidden = torch.zeros(rows * steps, 2)

    def make_spec(enabled):
        spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
        spec.num_speculative_steps = steps
        spec._draft_topk = None
        spec.use_confidence_head = False
        spec.draft_logits = None
        spec._step_cols = torch.arange(steps)
        spec.sample_indices = torch.arange(rows * steps)
        spec.sample_idx_mapping = torch.tensor([4, 1, -1]).repeat_interleave(steps)
        spec.sample_pos = torch.arange(rows * steps)
        spec._anchor_idx = torch.arange(rows) * steps
        ids = torch.zeros(rows * steps, dtype=torch.int64)
        ids[spec._anchor_idx] = torch.tensor([0, 3, 5])
        spec.input_buffers = SimpleNamespace(input_ids=ids)
        spec.draft_tokens = torch.empty(rows, steps, dtype=torch.int64)
        spec._run_model = lambda *args: sample_hidden
        spec._maybe_predict_acceptance = lambda *args: None
        spec.model = SimpleNamespace(
            compute_draft_logits=lambda hidden: torch.zeros(hidden.shape[0], vocab),
            markov_embed=lambda previous: torch.nn.functional.one_hot((previous + 1) % vocab, vocab).float(),
            markov_bias=lambda embedding: embedding * 20,
            map_draft_to_target=lambda ids: ids,
        )
        spec.spec_k_draft = SpecKDraftBudgetState(policy(), 5, steps, vocab) if enabled else None
        return spec

    baseline, dynamic = make_spec(False), make_spec(True)
    baseline._generate_draft(rows, rows * steps, None, None, None, CUDAGraphMode.FULL)
    dynamic._generate_draft(rows, rows * steps, None, None, None, CUDAGraphMode.FULL)
    assert torch.equal(dynamic.draft_tokens, baseline.draft_tokens)
    assert dynamic.draft_tokens[0].tolist() == [1, 2, 3, 4, 5, 6, 7]
    # Raw logits are uniform and would produce six experts; corrected logits
    # are sharply peaked and must produce four with these thresholds.
    assert dynamic.spec_k_draft.budgets[4].tolist() == [4] * steps + [6]
    assert dynamic.spec_k_draft.budgets[1].tolist() == [4] * steps + [6]
    assert torch.all(dynamic.spec_k_draft.budgets[[0, 2, 3]] == 6)


def test_indexed_pending_drafts_survive_unscheduled_steps_and_reordering():
    state = budget_state()
    cache = torch.full((4, 8), 6, dtype=torch.int32)
    cache[3] = torch.tensor([3] * 7 + [6])
    cache[1] = torch.tensor([4] * 7 + [6])
    state.set_indexed_drafts(["a"], [3], cache)
    state.set_indexed_drafts(["b"], [1], cache)
    actual = state.prepare(
        ["b", "a"], np.array([8, 8]), {"a": [1] * 7, "b": [1] * 7}, [4, 4], torch.arange(4, 12).repeat(2), 24
    )
    assert actual[:16].tolist() == [4] * 7 + [6] + [3] * 7 + [6]
    assert actual[16:].tolist() == [6] * 8
    # Preemption invalidates its freed V2 request slot, but used history stays.
    state.invalidate_drafts(["a"])
    cache[3].fill_(5)
    recompute = state.prepare(["a"], np.array([4]), {}, [4], torch.arange(4, 8), 8)
    assert recompute[:4].tolist() == [3] * 4
    state.release(["a"])
    assert "a" not in state.previous_rows and "b" in state.previous_rows
    with pytest.raises(ValueError, match="invalid request slot"):
        state.set_indexed_drafts(["a"], [4], cache)


def test_runner_consumes_real_device_positions_and_original_prompt_lengths():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._spec_k_state = state = budget_state()
    runner.req_states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.array([2, 4, 6, 3])))
    cache = torch.full((4, 8), 6, dtype=torch.int32)
    cache[3] = torch.tensor([3] * 7 + [6])
    state.set_indexed_drafts(["a"], [3], cache)
    batch = SimpleNamespace(
        req_ids=["a"],
        idx_mapping_np=np.array([3]),
        num_scheduled_tokens=np.array([8]),
        positions=torch.arange(3, 19),
        num_tokens=8,
        num_tokens_after_padding=16,
    )
    scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"a": [1] * 7})
    runner._prepare_spec_k_inputs(batch, scheduler)
    assert state.input_top_ks[:16].tolist() == [3] * 7 + [6] * 9


def test_target_budget_context_is_scoped_and_retains_fixed_storage():
    first = torch.full((16,), 3, dtype=torch.int32)
    second = torch.full((16,), 5, dtype=torch.int32)
    assert get_mrv2_token_top_ks(8) is None
    with override_mrv2_token_top_ks(first):
        assert get_mrv2_token_top_ks(8).data_ptr() == first.data_ptr()
        with override_mrv2_token_top_ks(second):
            assert get_mrv2_token_top_ks(8).tolist() == [5] * 8
        assert get_mrv2_token_top_ks(8).tolist() == [3] * 8
        with pytest.raises(ValueError, match="exceeds"):
            get_mrv2_token_top_ks(17)
        with pytest.raises(RuntimeError), override_mrv2_token_top_ks(None):
            assert get_mrv2_token_top_ks(8) is None
            raise RuntimeError("failure")
        assert get_mrv2_token_top_ks(8).tolist() == [3] * 8
    assert get_mrv2_token_top_ks(8) is None


@pytest.mark.parametrize("dummy,profile", [(False, False), (True, False), (False, True)])
def test_propose_publishes_host_mapping_after_replay_only_for_real_requests(monkeypatch, dummy, profile):
    import vllm_ascend.worker.v2.spec_decode.dspark.speculator as module

    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.replicated_pcp = spec.enable_adaptive_verification = spec.use_dcp = False
    spec.vllm_config = SimpleNamespace(parallel_config=object())
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *args, **kwargs: nullcontext())
    spec.input_buffers = SimpleNamespace(positions=object())
    spec.max_num_tokens = 64
    spec.spec_k_target = Mock()
    spec.spec_k_draft = SimpleNamespace(budgets=torch.full((4, 8), 6, dtype=torch.int32))
    output = torch.ones(2, 7, dtype=torch.int64)
    monkeypatch.setattr(DSparkSpeculator, "propose", lambda *args, **kwargs: output)
    batch = SimpleNamespace(
        req_ids=["b", "a"], idx_mapping_np=np.array([1, 3]), is_prefilling_np=np.array([False, False])
    )
    assert (
        spec.propose(batch, {}, {}, None, None, None, None, None, None, None, None, dummy_run=dummy, is_profile=profile)
        is output
    )
    if dummy or profile:
        spec.spec_k_target.set_indexed_drafts.assert_not_called()
    else:
        call = spec.spec_k_target.set_indexed_drafts.call_args.args
        assert call[0] == ["b", "a"]
        assert np.array_equal(call[1], [1, 3])
        assert call[2] is spec.spec_k_draft.budgets


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_runner_initializes_shared_policy_and_tp0_only_statistics(monkeypatch, tp_rank):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    p = policy()
    runner.ascend_config = SimpleNamespace(spec_k_config=SpecKConfig(enabled=True, ppl_thresholds=(3.1469, 1.8222)))
    runner.speculator = SimpleNamespace(spec_k_policy=p)
    runner.max_num_reqs = 4
    runner.max_model_len = 64
    runner.max_num_tokens = 64
    runner.device = torch.device("cpu")
    runner.parallel_config = SimpleNamespace(data_parallel_rank=2)
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.get_tensor_model_parallel_rank", lambda: tp_rank)
    runner._init_spec_k()
    assert runner._spec_k_state.policy is p
    assert runner.speculator.spec_k_target is runner._spec_k_state
    assert (runner._spec_k_state.stats is not None) == (tp_rank == 0)
    assert not runner._spec_k_state.metadata.cpu.is_pinned()
    assert runner._spec_k_state.contiguous_positions
