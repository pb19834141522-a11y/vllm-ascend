# SPDX-License-Identifier: Apache-2.0
"""Mixed target batches must reuse the steady-decode route kernel."""

import torch

from vllm_ascend.ops.triton.spec_decode.spec_k import _prune_routes, prune_routes


def test_route_row_count_does_not_create_new_compiled_variants():
    # Separate buffers keep identical pointer alignment and strides. Include
    # N=1 and multiples of 16 to catch value AND alignment specialization.
    variants = None
    for rows in (1, 16, 32, 33, 79, 137, 233, 240, 256):
        weights = torch.ones((rows, 6), dtype=torch.float32, device="npu")
        ids = torch.arange(6, dtype=torch.int32, device="npu").repeat(rows, 1)
        budgets = torch.arange(rows, dtype=torch.int32, device="npu") % 7
        prune_routes(weights, ids, budgets, 384)
        torch.npu.synchronize()
        active = torch.arange(6)[None, :] < (torch.arange(rows) % 7)[:, None]
        assert torch.equal(ids.cpu(), torch.where(active, torch.arange(6), 384).int())
        assert torch.equal(weights.cpu(), active.float())
        if hasattr(_prune_routes, "device_caches"):
            count = sum(len(entry[0]) for entry in _prune_routes.device_caches.values())
        else:
            count = sum(len(cache) for cache in _prune_routes.cache.values())
        if variants is None:
            variants = count
        assert count == variants, f"New route-kernel specialization for {rows=}"
