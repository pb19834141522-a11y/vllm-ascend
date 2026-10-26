# SPDX-License-Identifier: Apache-2.0
"""V2 entropy budgets must follow request slots on every NPU graph replay."""

from functools import partial

import numpy as np
import pytest
import torch
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.spec_k import SpecKBudgetState, SpecKDraftBudgetState, SpecKPolicy, prune_fused_routes
from vllm_ascend.spec_k_config import SpecKConfig
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("apply_bonus", [False, True])
def test_post_markov_budget_graph_replays_changed_logits_and_request_mapping(steps, apply_bonus):
    config = SpecKConfig(enabled=True, ppl_thresholds=(3.1469, 1.8222), apply_last_token=apply_bonus)
    policy = SpecKPolicy(config, 6, torch.device("npu"))
    reference = SpecKPolicy(config, 6, torch.device("cpu"))
    state = SpecKDraftBudgetState(policy, 8, steps, 129280)
    logits = torch.zeros((steps, 3, 129280), device="npu")
    logits[:, 0, 0] = 30
    indices = torch.tensor([6, 1, -1], device="npu", dtype=torch.int32).repeat_interleave(steps)
    pointer = state.budgets.data_ptr()

    def forward():
        for step in range(steps):
            state.record(logits[step], step)
        state.finish(3, indices)

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()
    for mapping in [[3, 6, -1], [2, 3, -1]]:
        before = state.budgets.cpu()
        logits.zero_()
        logits[:, 1, 5] = 30
        indices.copy_(torch.tensor(mapping, device="npu", dtype=torch.int32).repeat_interleave(steps))
        graph.replay()
        torch.npu.synchronize()
        expected = reference.top_ks_from_logits(logits.permute(1, 0, 2).cpu())
        for row, slot in enumerate(mapping):
            if slot >= 0:
                before[slot] = expected[row]
        assert torch.equal(state.budgets.cpu(), before)
        assert state.budgets.data_ptr() == pointer


@pytest.mark.parametrize("collect_histogram", [False, True])
@torch.inference_mode()
def test_budget_preparation_between_graphs_without_per_step_synchronization(collect_histogram):
    """The old History scatter's software lock stalls this interleaving."""
    enable_custom_op()
    device = torch.device("npu")
    requests, steps, vocab, iterations = 8, 7, 129280, 160
    tokens = requests * (steps + 1)
    policy = SpecKPolicy(SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128)), 6, device)
    draft = SpecKDraftBudgetState(policy, requests, steps, vocab)
    target = SpecKBudgetState(
        policy,
        requests,
        1024,
        tokens,
        partial(CpuGpuBuffer, device=device, pin_memory=False),
        log_stats_interval=100 if collect_histogram else 0,
        contiguous_positions=True,
    )
    logits = torch.zeros((steps, requests, vocab), device=device)
    mapping = torch.arange(requests, dtype=torch.int32, device=device).repeat_interleave(steps)
    positions = torch.arange(100, 108, device=device).repeat(requests)
    req_ids = [str(i) for i in range(requests)]
    counts = np.full(requests, steps + 1, dtype=np.int32)
    scheduled = {req_id: [0] * steps for req_id in req_ids}
    target.set_indexed_drafts(req_ids, list(range(requests)), draft.budgets)
    source_ids = torch.arange(6, dtype=torch.int32, device=device).repeat(tokens, 1)
    ids = torch.empty_like(source_ids)
    weights = torch.empty((tokens, 6), device=device)

    def draft_forward():
        for step in range(steps):
            draft.record(logits[step], step)
        draft.finish(requests, mapping)

    def prepare():
        target.prepare(req_ids, counts, scheduled, [100] * requests, positions, tokens)

    def target_forward():
        ids.copy_(source_ids)
        weights.fill_(1)
        prune_fused_routes(weights, ids, target.input_top_ks, 384)

    draft_forward()
    prepare()
    target_forward()
    torch.npu.synchronize()
    draft_graph, target_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(draft_graph):
        draft_forward()
    with torch.npu.graph(target_graph):
        target_forward()
    if target.stats is not None:
        target.stats.histogram.zero_()
    observed = torch.empty((iterations, tokens), dtype=torch.int32, device=device)
    weight_sums = torch.empty(iterations, device=device)
    for iteration in range(iterations):
        logits.zero_()
        if iteration % 2:
            logits[:, :, 0].fill_(30)
        draft_graph.replay()
        prepare()
        target_graph.replay()
        observed[iteration].copy_(target.input_top_ks)
        weight_sums[iteration] = weights.sum()
    # Synchronize only after all replays: per-step waits conceal the regression.
    torch.npu.synchronize()
    expected = torch.full((iterations, tokens), 6, dtype=torch.int32)
    expected[1::2] = torch.tensor([3] * steps + [6]).repeat(requests)
    assert torch.equal(observed.cpu(), expected)
    assert torch.equal(weight_sums.cpu(), expected.sum(dim=1).float())
    if target.stats is not None:
        assert target.stats.histogram.cpu().tolist() == [
            0,
            0,
            0,
            iterations // 2 * requests * steps,
            0,
            0,
            iterations // 2 * (tokens + requests),
        ]
        target.stats.close()
