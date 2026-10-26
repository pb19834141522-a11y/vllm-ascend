# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton

_ENTROPY_BLOCK = 4096
_ROUTE_BLOCK = 256
_BUDGET_BLOCK = 128
_METADATA_WIDTH = 7


@triton.jit
def _entropy_partials(
    Logits,
    Parts,
    V: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr = 1,
    STEP: tl.constexpr = 0,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Logits + row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
    maximum = tl.max(values, 0)
    finite = (values > -float("inf")) & (values < float("inf"))
    shifted = tl.where(finite, values - maximum, 0.0)
    exponentials = tl.where(finite, tl.exp(shifted), 0.0)
    invalid = tl.sum(((values != values) | (values == float("inf"))).to(tl.int32), 0) > 0
    offset = ((row * STEPS + STEP) * CHUNKS + chunk) * 3
    tl.store(Parts + offset, tl.where(invalid, float("nan"), maximum))
    tl.store(Parts + offset + 1, tl.sum(exponentials, 0))
    tl.store(Parts + offset + 2, tl.sum(exponentials * shifted, 0))


@triton.jit
def _entropy_budgets(
    Parts,
    Thresholds,
    Budgets,
    S: tl.constexpr,
    CHUNKS: tl.constexpr,
    N_THRESHOLDS: tl.constexpr,
    BASE_K: tl.constexpr,
    APPLY_BONUS: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    CHUNK_BLOCK: tl.constexpr,
    THRESHOLD_BLOCK: tl.constexpr,
    RowIndices,
    ROW_STRIDE: tl.constexpr,
    OUTPUT_ROWS: tl.constexpr,
):
    req = tl.program_id(0)
    output_row = req
    if ROW_STRIDE:
        output_row = tl.load(RowIndices + req * ROW_STRIDE)
    # Skip padded request slots before any entropy read or output write.
    if (output_row >= 0) & (output_row < OUTPUT_ROWS):
        tokens = tl.arange(0, TOKEN_BLOCK)
        chunks = tl.arange(0, CHUNK_BLOCK)
        mask = (tokens[:, None] < S) & (chunks[None, :] < CHUNKS)
        offsets = ((req * S + tokens[:, None]) * CHUNKS + chunks[None, :]) * 3
        maxima = tl.load(Parts + offsets, mask, other=-float("inf"))
        sums = tl.load(Parts + offsets + 1, mask, other=0.0)
        weighted = tl.load(Parts + offsets + 2, mask, other=0.0)
        maximum = tl.max(maxima, 1)
        valid = mask & (sums > 0) & (maxima == maxima)
        shifts = tl.where(valid, maxima - maximum[:, None], 0.0)
        factors = tl.where(valid, tl.exp(shifts), 0.0)
        total = tl.sum(factors * sums, 1)
        moment = tl.sum(factors * (weighted + sums * shifts), 1)
        entropy = tl.log(total) - moment / total
        invalid = (tl.sum((maxima != maxima).to(tl.int32), 1) > 0) | (total <= 0) | (entropy != entropy)
        entropy = tl.maximum(entropy, 0.0)
        threshold_slots = tl.arange(0, THRESHOLD_BLOCK)
        thresholds = tl.load(Thresholds + threshold_slots, threshold_slots < N_THRESHOLDS, other=float("inf"))
        increments = tl.sum(
            ((entropy[:, None] >= thresholds[None, :]) & (threshold_slots[None, :] < N_THRESHOLDS)).to(tl.int32), 1
        )
        budgets = tl.where(invalid, BASE_K, BASE_K - N_THRESHOLDS + increments)
        tl.store(Budgets + output_row * (S + 1) + tokens, budgets, tokens < S)
        bonus = BASE_K
        if APPLY_BONUS:
            bonus = tl.sum(tl.where(tokens < S, budgets, 0), 0) // S
        tl.store(Budgets + output_row * (S + 1) + S, bonus)


def create_entropy_partials(batch, steps, vocab, device):
    return torch.empty((batch * steps, triton.cdiv(vocab, _ENTROPY_BLOCK), 3), device=device, dtype=torch.float32)


def finish_entropy_budgets(parts, thresholds, budgets, rows, steps, base_top_k, apply_last_token, row_indices=None):
    if rows:
        _entropy_budgets[(rows,)](
            parts,
            thresholds,
            budgets,
            steps,
            parts.shape[1],
            thresholds.numel(),
            base_top_k,
            apply_last_token,
            triton.next_power_of_2(steps),
            triton.next_power_of_2(parts.shape[1]),
            triton.next_power_of_2(max(1, thresholds.numel())),
            budgets if row_indices is None else row_indices,
            0 if row_indices is None else steps,
            budgets.shape[0],
        )


def entropy_top_ks(logits, thresholds, base_top_k, apply_last_token):
    logits = logits.contiguous()
    batch, tokens, vocab = logits.shape
    if batch == 0:
        return torch.empty((batch, tokens + 1), device=logits.device, dtype=torch.int32)
    chunks = triton.cdiv(vocab, _ENTROPY_BLOCK)
    parts = torch.empty((batch * tokens, chunks, 3), device=logits.device, dtype=torch.float32)
    budgets = torch.empty((batch, tokens + 1), device=logits.device, dtype=torch.int32)
    _entropy_partials[(batch * tokens, chunks)](
        logits,
        parts,
        vocab,
        chunks,
        _ENTROPY_BLOCK,
    )
    finish_entropy_budgets(parts, thresholds, budgets, batch, tokens, base_top_k, apply_last_token)
    return budgets


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def _prune_routes(
    Weights,
    Ids,
    Budgets,
    N,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, slot = index // K, index % K
    budget = tl.load(Budgets + row, row < N, other=K)
    drop = (row < N) & (slot >= budget)
    tl.store(Ids + row * I0 + slot * I1, EXPERTS, drop)
    tl.store(Weights + row * W0 + slot * W1, 0.0, drop)


def prune_routes(weights, ids, budgets, num_experts):
    rows, width = ids.shape
    # Replacement prefills and mixed batches have many different row counts.
    # N must stay a runtime bound, including its alignment specialization, so
    # those batches reuse the kernel captured for steady decode.
    if rows == 0:
        return
    _prune_routes[(triton.cdiv(rows * width, _ROUTE_BLOCK),)](
        weights,
        ids,
        budgets,
        rows,
        width,
        num_experts,
        weights.stride(0),
        weights.stride(1),
        ids.stride(0),
        ids.stride(1),
        _ROUTE_BLOCK,
    )


@triton.jit
def _prepare_budgets(
    History,
    Out,
    Previous,
    Metadata,
    Positions,
    Histogram,
    MAX_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    BASE_K: tl.constexpr,
    META_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_STATS: tl.constexpr,
    CONTIGUOUS_POSITIONS: tl.constexpr,
):
    req = tl.program_id(0)
    tile = tl.program_id(1)
    slot = tl.load(Metadata + req * META_WIDTH)
    previous_row = tl.load(Metadata + req * META_WIDTH + 1)
    span = tl.load(Metadata + req * META_WIDTH + 2)
    start = tl.load(Metadata + req * META_WIDTH + 3)
    count = tl.load(Metadata + req * META_WIDTH + 4)
    prompt_len = tl.load(Metadata + req * META_WIDTH + 5)
    pending_width = tl.load(Metadata + req * META_WIDTH + 6)
    local = tile * BLOCK + tl.arange(0, BLOCK)
    active = local < count
    if CONTIGUOUS_POSITIONS:
        # V1 emits consecutive positions within each scheduled request. Keep
        # that affine address visible to the compiler: a gathered scatter to
        # History otherwise generates software locks that can stall when an
        # eager budget kernel interleaves with target/draft graph replays.
        position_start = tl.load(Positions + start + tile * BLOCK, tile * BLOCK < count, other=0).to(tl.int64)
        position = position_start + tl.arange(0, BLOCK)
    else:
        position = tl.load(Positions + start + local, active, other=-1).to(tl.int64)
    valid_position = active & (position >= 0) & (position < MAX_LEN)
    history_offset = slot.to(tl.int64) * MAX_LEN + position
    budgets = tl.load(History + history_offset, valid_position, other=BASE_K)
    in_span = active & (span > 0) & (local >= count - span) & (previous_row >= 0)
    # Preserve the configured bonus budget when scheduling truncates drafts.
    source = previous_row * WIDTH + tl.where(local == count - 1, pending_width - 1, local - (count - span))
    proposed = tl.load(Previous + source, in_span, other=BASE_K)
    budgets = tl.where(in_span, proposed, budgets)
    budgets = tl.where(position < prompt_len, BASE_K, budgets)
    tl.store(Out + start + local, budgets, active)
    tl.store(History + history_offset, budgets, valid_position)
    if HAS_STATS:
        # Count actual target generation-position executions, including rejected
        # speculative rows/bonus; prompt positions and graph padding never count.
        generation = valid_position & (position >= prompt_len)
        for k in tl.static_range(1, BASE_K + 1):
            count_k = tl.sum((generation & (budgets == k)).to(tl.int32), 0)
            tl.atomic_add(Histogram + k, count_k, mask=count_k > 0, sem="relaxed")


def prepare_budgets(
    history,
    output,
    previous,
    metadata,
    positions,
    max_len,
    max_count,
    base_top_k,
    histogram=None,
    *,
    contiguous_positions=False,
):
    width = 0 if previous is None else previous.shape[1]
    _prepare_budgets[(metadata.shape[0], triton.cdiv(max_count, _BUDGET_BLOCK))](
        history,
        output,
        output if previous is None else previous,
        metadata,
        positions,
        output if histogram is None else histogram,
        max_len,
        width,
        base_top_k,
        _METADATA_WIDTH,
        _BUDGET_BLOCK,
        histogram is not None,
        contiguous_positions,
    )
