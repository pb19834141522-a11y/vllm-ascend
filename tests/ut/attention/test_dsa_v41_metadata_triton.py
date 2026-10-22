# SPDX-License-Identifier: Apache-2.0
"""Check the shared A3/A5 metadata kernels against tensor semantics on CPU."""

import os

import pytest
import torch

from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata
from vllm_ascend.ops.triton.packed_cache_slot_mapping import build_packed_cache_slot_mapping

pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="Requires CPU Triton interpreter")


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("page_size", [64, 128])
def test_slot_coordinates_preserve_physical_groups_and_mask_stale_padding(ratio, skip, page_size):
    # Non-contiguous inputs, negative slots, incomplete C2 groups, and positive
    # stale graph padding. Different cache groups keep their own physical IDs.
    slots = torch.tensor([255, 0, -1, 0, 258, 0, 259, 0, 511, 0, 513, 0, 515, 0])[::2]
    positions = torch.tensor([3, 0, 4, 0, 6, 0, 7, 0, 9, 0, 11, 0, 13, 0])[::2]
    offsets = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    coords = torch.empty((7, 2), dtype=torch.int32)
    flat = torch.empty(7, dtype=torch.int64)
    build_packed_cache_slot_mapping(
        slots,
        positions,
        offsets,
        7,
        2,
        6,
        page_size,
        ratio,
        skip_update=skip,
        coordinates_output=coords,
        flat_output=flat,
    )
    valid = (slots >= 0) & (torch.arange(7) < 5)
    physical = slots
    if ratio == 2:
        valid &= ((slots + 1) % 2 == 0) & (positions % 2 == 1) & (not skip)
        physical = slots // 2
    expected = torch.where(valid, physical, -1)
    assert torch.equal(flat, expected)
    assert torch.equal(coords[:, 0], torch.where(valid, physical.clamp_min(0) // page_size, -1).int())
    assert torch.equal(coords[:, 1], torch.where(valid, physical.clamp_min(0) % page_size, -1).int())


@pytest.mark.parametrize("skip", [False, True])
def test_ring_controls_and_source_rope_match_eager_metadata(skip):
    starts = torch.tensor([0, 3, 6, 8], dtype=torch.int32)
    lengths = torch.tensor([9, 12, 50], dtype=torch.int32)
    positions = torch.tensor([6, 7, 8, 9, 10, 11, 40, 41], dtype=torch.int64)
    # Use a strided first-column block-table view, as the real builder does.
    blocks = torch.tensor([[4, 99, 5], [6, 99, 7], [8, 99, 9]], dtype=torch.int32)[:, ::2]
    cos = torch.arange(64 * 8, dtype=torch.float32).view(64, 1, 1, 8)
    sin = -cos
    ring = torch.empty((5, 3), dtype=torch.int32)
    complete = torch.empty(8, dtype=torch.bool)
    source = torch.empty(8, dtype=torch.int64)
    out_cos, out_sin = torch.empty((8, 1, 1, 8)), torch.empty((8, 1, 1, 8))
    build_c2_ring_metadata(
        starts,
        lengths,
        positions,
        blocks,
        cos,
        sin,
        3,
        8,
        2,
        5,
        skip_update=skip,
        ring_metadata_output=ring,
        complete_mask_output=complete,
        source_positions_output=source,
        cos_output=out_cos,
        sin_output=out_sin,
    )
    used = (starts[1:].clamp_max(5) - starts[:-1]).clamp_min(0)
    used[2:] = 0
    if skip:
        used.zero_()
    expected_ring = torch.stack(
        (
            (lengths - starts.diff()).clamp_min(0),
            used,
            starts[:-1],
            starts[:-1],
            torch.where(used > 0, blocks[:, 0], 0),
        )
    )
    expected_complete = (positions % 2 == 1) & (torch.arange(8) < 5) & (not skip)
    expected_source = torch.where(expected_complete, positions - 1, 0)
    assert torch.equal(ring, expected_ring)
    assert torch.equal(complete, expected_complete)
    assert torch.equal(source, expected_source)
    assert torch.equal(out_cos, cos[expected_source])
    assert torch.equal(out_sin, sin[expected_source])
