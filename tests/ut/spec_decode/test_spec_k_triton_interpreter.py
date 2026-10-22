# SPDX-License-Identifier: Apache-2.0
"""Execute the real Triton source in its CPU interpreter, without any NPU."""

import os

import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.spec_k import entropy_top_ks, prepare_budgets, prune_routes
from vllm_ascend.spec_k import SpecKPolicy
from vllm_ascend.spec_k_config import SpecKConfig

pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="Requires CPU Triton interpreter")


@pytest.fixture(autouse=True)
def forbid_npu_initialization(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Triton interpreter validation must stay on CPU")

    monkeypatch.setattr(torch.npu, "_lazy_init", fail)


@pytest.mark.parametrize("vocab", [1, 129, 4095, 4096, 4097, 129280])
@pytest.mark.parametrize("apply_bonus", [False, True])
def test_real_entropy_kernel_matches_original_policy(vocab, apply_bonus):
    generator = torch.Generator().manual_seed(314)
    logits = torch.randn(2, 5, vocab, generator=generator) * 3
    config = SpecKConfig(enabled=True, ppl_thresholds=(3.1469, 1.8222), apply_last_token=apply_bonus)
    policy = SpecKPolicy(config, 6, torch.device("cpu"))
    expected = policy.top_ks_from_logits(logits)
    actual = entropy_top_ks(logits, policy.log_thresholds, 6, apply_bonus)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("drafts", [1, 7])
def test_real_entropy_kernel_accepts_configurable_draft_length(drafts):
    logits = torch.zeros(2, drafts, 4097)
    logits[:, :, 8:] = -torch.inf
    policy = SpecKPolicy(SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128)), 6, torch.device("cpu"))
    actual = entropy_top_ks(logits, policy.log_thresholds, 6, False)
    assert actual.tolist() == [[3] * drafts + [6]] * 2


@pytest.mark.parametrize("thresholds", [(1.0,), (128, 128, 128)])
def test_masked_vocab_and_invalid_distributions(thresholds):
    logits = torch.full((2, 5, 4097), -torch.inf)
    logits[:, :, :2] = 0
    logits[0, 0] = -torch.inf
    logits[0, 1, 0] = torch.nan
    logits[1, 2, 0] = torch.inf
    policy = SpecKPolicy(SpecKConfig(enabled=True, ppl_thresholds=thresholds), 6, torch.device("cpu"))
    assert torch.equal(entropy_top_ks(logits, policy.log_thresholds, 6, False), policy.top_ks_from_logits(logits))


def test_route_kernel_handles_strided_views_and_keeps_real_experts_unchanged():
    weights = torch.ones(7, 12)[:, ::2]
    ids = torch.arange(12, dtype=torch.int32).repeat(7, 1)[:, ::2]
    original = ids.clone()
    budgets = torch.arange(7, dtype=torch.int32)
    prune_routes(weights, ids, budgets, 384)
    active = torch.arange(6)[None, :] < budgets[:, None]
    assert torch.equal(ids[active], original[active])
    assert torch.all(ids[~active] == 384) and torch.all(weights[~active] == 0)


@pytest.mark.parametrize("rows", [0, 1, 16, 32, 33, 79, 137, 233, 240, 256])
def test_route_kernel_handles_replacement_prefill_row_counts(rows):
    weights = torch.rand(rows, 6)
    ids = torch.arange(6, dtype=torch.int32).repeat(rows, 1)
    original_weights = weights.clone()
    original_ids = ids.clone()
    budgets = torch.arange(rows, dtype=torch.int32) % 7
    prune_routes(weights, ids, budgets, 384)
    active = torch.arange(6)[None, :] < budgets[:, None]
    assert torch.equal(ids[active], original_ids[active])
    assert torch.equal(weights[active], original_weights[active])
    assert torch.all(ids[~active] == 384) and torch.all(weights[~active] == 0)


def test_real_metadata_kernel_reorders_recomputes_and_preserves_prompt_rows():
    history = torch.full((3, 64), 6, dtype=torch.int32)
    history[0, 12:15] = torch.tensor([5, 4, 3])
    previous = torch.tensor([[3, 4, 5, 3, 4, 6], [5, 3, 4, 5, 3, 6]], dtype=torch.int32)
    # slot, previous proposer row, speculative span, target offset/count, prompt length.
    metadata = torch.tensor([[1, 1, 6, 0, 6, 5], [0, -1, 0, 6, 3, 3], [2, 0, 3, 9, 3, 1]], dtype=torch.int32)
    positions = torch.tensor([5, 6, 7, 8, 9, 10, 12, 13, 14, 0, 1, 2], dtype=torch.int64)
    actual_history = history.clone()
    actual = torch.full((16,), 6, dtype=torch.int32)
    prepare_budgets(actual_history, actual, previous, metadata, positions, 64, 6, 6)
    assert actual.tolist() == [5, 3, 4, 5, 3, 6, 5, 4, 3, 6, 4, 5, 6, 6, 6, 6]
    assert torch.equal(actual_history[1, 5:11], previous[1])
    assert actual_history[2, 0].item() == 6


def test_real_metadata_kernel_spans_multiple_tiles_without_previous_drafts():
    history = torch.full((1, 512), 6, dtype=torch.int32)
    history[0, 200:400] = 3
    output = torch.full((512,), 6, dtype=torch.int32)
    metadata = torch.tensor([[0, -1, 0, 0, 400, 100]], dtype=torch.int32)
    positions = torch.arange(400, dtype=torch.int32)
    prepare_budgets(history, output, None, metadata, positions, 512, 400, 6)
    assert torch.all(output[:200] == 6)
    assert torch.all(output[200:400] == 3)
    assert torch.all(output[400:] == 6)


def test_real_budget_kernel_counts_consumed_budgets_excluding_prompt_and_padding():
    history = torch.full((2, 512), 6, dtype=torch.int32)
    history[0, 200:400] = 3
    previous = torch.tensor([[3] * 7 + [6]], dtype=torch.int32)
    # a spans several programs; b verifies eight rows, two of them prompt.
    metadata = torch.tensor([[0, -1, 0, 0, 400, 100], [1, 0, 8, 400, 8, 2]], dtype=torch.int32)
    positions = torch.cat((torch.arange(400), torch.arange(8)))
    output = torch.full((512,), 6, dtype=torch.int32)
    histogram = torch.zeros(7, dtype=torch.int32)
    prepare_budgets(history, output, previous, metadata, positions, 512, 400, 6, histogram)
    assert histogram.tolist() == [0, 0, 0, 205, 0, 0, 101]
    # A second execution is real verification work even if positions repeat.
    prepare_budgets(history, output, previous, metadata, positions, 512, 400, 6, histogram)
    assert histogram.tolist() == [0, 0, 0, 410, 0, 0, 202]


@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("vocab", [129, 4097, 129280])
def test_v2_post_markov_partials_scatter_by_request_slot(steps, vocab):
    from vllm_ascend.ops.triton.spec_decode.spec_k import (
        create_entropy_partials,
        finish_entropy_budgets,
        record_entropy_partials,
    )

    policy = SpecKPolicy(SpecKConfig(enabled=True, ppl_thresholds=(3.1469, 1.8222)), 6, torch.device("cpu"))
    logits = torch.zeros(3, steps, vocab)
    logits[0, :, 0] = 30
    logits[2, :, 0] = 40
    cache = torch.full((5, steps + 1), 6, dtype=torch.int32)
    parts = create_entropy_partials(5, steps, vocab, torch.device("cpu"))
    mapping = torch.tensor([4, 1, -1], dtype=torch.int32).repeat_interleave(steps)
    for step in range(steps):
        record_entropy_partials(logits[:, step], parts, steps, step)
    finish_entropy_budgets(parts, policy.log_thresholds, cache, 3, steps, 6, False, mapping)
    expected = policy.top_ks_from_logits(logits)
    assert torch.equal(cache[4], expected[0])
    assert torch.equal(cache[1], expected[1])
    assert torch.all(cache[[0, 2, 3]] == 6)
    # Real-request batch shrinks/reorders; the absent request keeps its budget.
    for step in range(steps):
        record_entropy_partials(torch.zeros(1, vocab), parts, steps, step)
    finish_entropy_budgets(
        parts,
        policy.log_thresholds,
        cache,
        1,
        steps,
        6,
        False,
        torch.tensor([2], dtype=torch.int32).repeat_interleave(steps),
    )
    assert torch.equal(cache[4], expected[0])
