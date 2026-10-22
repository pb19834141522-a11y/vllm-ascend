# SPDX-License-Identifier: Apache-2.0
"""V2 entropy budgets must follow request slots on every NPU graph replay."""

import pytest
import torch

from vllm_ascend.spec_k import SpecKDraftBudgetState, SpecKPolicy
from vllm_ascend.spec_k_config import SpecKConfig


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
