# SPDX-License-Identifier: Apache-2.0

from collections import deque
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from vllm_ascend.spec_k_config import SpecKConfig
from vllm_ascend.spec_k_stats import SpecKStats

SPEC_K_METADATA_WIDTH = 6


class SpecKPolicy:
    """Preserve the original post-Markov DSpark perplexity-to-budget policy."""

    def __init__(self, config: SpecKConfig, base_top_k: int, device: torch.device):
        if base_top_k <= len(config.ppl_thresholds):
            raise ValueError("Spec-K needs fewer thresholds than the target model top-k")
        self.base_top_k = base_top_k
        self.apply_last_token = config.apply_last_token
        self.log_thresholds = torch.tensor(config.ppl_thresholds, dtype=torch.float32, device=device).log_()

    def top_ks_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("Spec-K draft logits must have shape [requests, drafts, vocabulary]")
        if logits.device.type == "npu":
            # Lazy loading keeps CPU policy/state tests independent of the NPU compiler.
            from vllm_ascend.ops.triton.spec_decode.spec_k import entropy_top_ks

            return entropy_top_ks(logits, self.log_thresholds, self.base_top_k, self.apply_last_token)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        entropy = -torch.xlogy(probabilities, probabilities).sum(dim=-1)
        budgets = (
            self.base_top_k - self.log_thresholds.numel() + (entropy[..., None] >= self.log_thresholds).sum(dim=-1)
        )
        # Invalid distributions must never accidentally prune every route.
        budgets = torch.where(torch.isfinite(entropy), budgets, self.base_top_k).to(torch.int32)
        bonus = (
            budgets.float().mean(dim=1).to(torch.int32)
            if self.apply_last_token
            else torch.full((logits.shape[0],), self.base_top_k, dtype=torch.int32, device=logits.device)
        )
        return torch.cat((budgets, bonus[:, None]), dim=1).contiguous()


class SpecKDraftBudgetState:
    """Capture post-Markov entropy into fixed V2 request-slot budget buffers.

    Each sampling hook records only small entropy partials on NPU, not a
    second full-vocabulary logits tensor. The existing draft graph captures
    these operations. Final budgets scatter directly by the speculator's
    persistent sample_idx_mapping; unscheduled requests retain their budgets.
    """

    def __init__(self, policy: SpecKPolicy, max_num_reqs: int, steps: int, vocab_size: int):
        self.policy = policy
        self.steps = steps
        self.vocab_size = vocab_size
        device = policy.log_thresholds.device
        self.budgets = torch.full((max_num_reqs, steps + 1), policy.base_top_k, dtype=torch.int32, device=device)
        self.parts = None
        self.logits = None
        if device.type == "npu":
            from vllm_ascend.ops.triton.spec_decode.spec_k import create_entropy_partials

            self.parts = create_entropy_partials(max_num_reqs, steps, vocab_size, device)
        else:
            self.logits = torch.empty((max_num_reqs, steps, vocab_size), device=device, dtype=torch.float32)

    def record(self, logits: torch.Tensor, step: int) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.vocab_size or logits.shape[0] > self.budgets.shape[0]:
            raise ValueError("Spec-K post-Markov logits do not match the allocated draft capacity")
        if not 0 <= step < self.steps:
            raise ValueError("Spec-K draft step is outside the fixed verification block")
        if self.parts is not None:
            from vllm_ascend.ops.triton.spec_decode.spec_k import record_entropy_partials

            record_entropy_partials(logits, self.parts, self.steps, step)
        else:
            self.logits[: logits.shape[0], step].copy_(logits)

    def finish(self, num_reqs: int, sample_idx_mapping: torch.Tensor) -> None:
        if not 0 <= num_reqs <= self.budgets.shape[0] or sample_idx_mapping.numel() < num_reqs * self.steps:
            raise ValueError("Spec-K draft request mapping exceeds the allocated capacity")
        if not num_reqs:
            return
        if self.parts is not None:
            from vllm_ascend.ops.triton.spec_decode.spec_k import finish_entropy_budgets

            finish_entropy_budgets(
                self.parts,
                self.policy.log_thresholds,
                self.budgets,
                num_reqs,
                self.steps,
                self.policy.base_top_k,
                self.policy.apply_last_token,
                sample_idx_mapping,
            )
        else:
            values = self.policy.top_ks_from_logits(self.logits[:num_reqs])
            rows = sample_idx_mapping[: num_reqs * self.steps : self.steps].long()
            valid = (rows >= 0) & (rows < self.budgets.shape[0])
            self.budgets.index_copy_(0, rows[valid], values[valid])


@torch.library.custom_op("vllm::ascend_spec_k_prune_routes", mutates_args=("weights", "ids"))
def prune_fused_routes(weights: torch.Tensor, ids: torch.Tensor, budgets: torch.Tensor, num_experts: int) -> None:
    """Use the fused kernel's reserved expert bucket, never a real EP expert.

    Width and allocations stay unchanged for graph replay. The reserved bucket
    is counted by init-routing but excluded from all real experts' receive/GMM
    counts. The patched unpermute skips zero-weight entries before reading them.
    Kept weights retain the original normalization/scaling.
    """
    if ids.ndim != 2 or weights.shape != ids.shape or budgets.shape != ids.shape[:1]:
        raise ValueError("Spec-K routing tensors must have matching token dimensions")
    if ids.device.type == "npu":
        from vllm_ascend.ops.triton.spec_decode.spec_k import prune_routes

        prune_routes(weights, ids, budgets, num_experts)
        return
    drop = torch.arange(ids.shape[1], device=ids.device)[None, :] >= budgets[:, None]
    ids.masked_fill_(drop, num_experts)
    weights.masked_fill_(drop, 0.0)


@prune_fused_routes.register_fake
def _prune_fused_routes_fake(weights, ids, budgets, num_experts):
    return None


def shard_token_budgets(
    budgets: torch.Tensor,
    local_tokens: int,
    tp_size: int,
    tp_rank: int,
    sequence_parallel: bool = False,
) -> torch.Tensor:
    """Match MC2's DP-wide padding followed by its true TP sequence shard."""
    if sequence_parallel:
        # SP splits each DP engine's own batch BEFORE padding its local shard
        # to the largest DP batch. Plain MC2 pads BEFORE the TP split.
        source_tokens = (budgets.numel() + tp_size - 1) // tp_size
        budgets = torch.nn.functional.pad(budgets, (0, source_tokens * tp_size - budgets.numel()), value=0)
        budgets = budgets[tp_rank * source_tokens : (tp_rank + 1) * source_tokens]
        if source_tokens > local_tokens:
            raise ValueError("Spec-K SP budgets exceed the routed local shard")
        return torch.nn.functional.pad(budgets, (0, local_tokens - source_tokens), value=0)
    padded_tokens = local_tokens * tp_size
    if budgets.numel() > padded_tokens:
        raise ValueError("Spec-K input budgets exceed the routed token span")
    if budgets.numel() < padded_tokens:
        budgets = torch.nn.functional.pad(budgets, (0, padded_tokens - budgets.numel()), value=0)
    return budgets[tp_rank * local_tokens : (tp_rank + 1) * local_tokens]


class SpecKBudgetState:
    """Keep budgets/history on device; CPU builds only per-request metadata.

    History stores the budget actually used at each target input position.
    Recording rejected rows is safe: the next verification overwrites them;
    historical recompute only addresses committed positions. Slots survive
    preemption and grow outside target graphs when necessary.
    """

    def __init__(
        self,
        policy: SpecKPolicy,
        max_num_reqs: int,
        max_model_len: int,
        max_num_tokens: int,
        make_buffer: Callable,
        log_stats_interval: int = 0,
        dp_rank: int = 0,
    ):
        self.policy = policy
        self.max_model_len = max_model_len
        device = policy.log_thresholds.device
        if device.type == "npu":
            try:
                abi = torch.ops._C_ascend.spec_k_fused_route_abi_version()
            except (AttributeError, RuntimeError) as error:
                raise RuntimeError(
                    "Spec-K needs the rebuilt vllm-ascend extension AND W4A8 fused custom-op package; "
                    "updating Python sources alone is unsafe. Rebuild with "
                    "pip install -v -e . --no-deps --no-build-isolation."
                ) from error
            if abi != 1:
                raise RuntimeError(f"Unsupported Spec-K fused-route ABI: {abi}")
        self.history = torch.full((max_num_reqs, max_model_len), policy.base_top_k, dtype=torch.int32, device=device)
        self.input_top_ks = torch.full((max_num_tokens,), policy.base_top_k, dtype=torch.int32, device=device)
        self.metadata = make_buffer(max_num_reqs, SPEC_K_METADATA_WIDTH, dtype=torch.int32)
        self.slots: dict[str, int] = {}
        self.free_slots = deque(range(max_num_reqs))
        self.previous_rows: dict[str, int] = {}
        self.previous_budgets: torch.Tensor | None = None
        self.stats = (
            SpecKStats(device, policy.base_top_k, log_stats_interval, max_num_tokens, dp_rank)
            if log_stats_interval
            else None
        )

    def release(self, finished_req_ids: Sequence[str]) -> None:
        had_requests = bool(self.slots)
        for req_id in finished_req_ids:
            slot = self.slots.pop(req_id, None)
            if slot is not None:
                self.free_slots.append(slot)
            self.previous_rows.pop(req_id, None)
        if self.stats is not None and had_requests and not self.slots:
            self.stats.flush("requests_drained", final=True)

    def _slot(self, req_id: str) -> int:
        if req_id in self.slots:
            return self.slots[req_id]
        if not self.slots and self.stats is not None:
            self.stats.start_session()
        if not self.free_slots:
            old_size = self.history.shape[0]
            extra = torch.full_like(self.history, self.policy.base_top_k)
            self.history = torch.cat((self.history, extra), dim=0)
            self.free_slots.extend(range(old_size, old_size * 2))
        slot = self.free_slots.popleft()
        self.history[slot].fill_(self.policy.base_top_k)
        self.slots[req_id] = slot
        return slot

    def set_drafts(self, req_ids: Sequence[str], budgets: torch.Tensor) -> None:
        if budgets.ndim != 2 or budgets.shape[0] != len(req_ids):
            raise ValueError("Spec-K draft budget rows do not match the proposer request batch")
        self.previous_rows = {req_id: row for row, req_id in enumerate(req_ids)}
        self.previous_budgets = budgets

    def set_indexed_drafts(self, req_ids: Sequence[str], row_indices: Sequence[int], budgets: torch.Tensor) -> None:
        """Bind V2's persistent request-slot output without copying budgets.

        Keep mappings for requests absent from this batch: their previously
        generated drafts remain in the scheduler and in their device slots.
        Preemption invalidates the slot mapping while preserving history.
        """
        if budgets.ndim != 2 or len(req_ids) != len(row_indices):
            raise ValueError("Spec-K indexed draft mapping has inconsistent dimensions")
        if any(not 0 <= int(row) < budgets.shape[0] for row in row_indices):
            raise ValueError("Spec-K indexed draft mapping contains an invalid request slot")
        if self.previous_budgets is not budgets:
            self.previous_rows.clear()
        self.previous_budgets = budgets
        self.previous_rows.update((req_id, int(row)) for req_id, row in zip(req_ids, row_indices))

    def invalidate_drafts(self, req_ids: Sequence[str]) -> None:
        for req_id in req_ids:
            self.previous_rows.pop(req_id, None)

    def prepare(
        self,
        req_ids: Sequence[str],
        counts: np.ndarray,
        scheduled_drafts: Mapping[str, Sequence[int]],
        num_prompt_tokens: Sequence[int],
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        if len(req_ids) != len(counts) or len(req_ids) > self.metadata.cpu.shape[0]:
            raise ValueError("Invalid Spec-K scheduler request metadata")
        if num_tokens_padded > self.input_top_ks.numel():
            raise ValueError("Spec-K input budget buffer is too small for this graph batch")
        width = 0 if self.previous_budgets is None else self.previous_budgets.shape[1]
        offset = 0
        for row, (req_id, count) in enumerate(zip(req_ids, counts)):
            previous_row = self.previous_rows.get(req_id, -1)
            drafts = len(scheduled_drafts.get(req_id, ()))
            span = min(drafts + 1, int(count), width) if drafts and previous_row >= 0 else 0
            self.metadata.np[row] = (self._slot(req_id), previous_row, span, offset, count, num_prompt_tokens[row])
            offset += int(count)
        result = self.input_top_ks[:num_tokens_padded]
        result.fill_(self.policy.base_top_k)
        if not req_ids:
            return result
        if positions.numel() < offset:
            raise ValueError("Spec-K target positions do not cover the scheduled rows")
        if result.device.type == "npu":
            from vllm_ascend.ops.triton.spec_decode.spec_k import prepare_budgets

            metadata = self.metadata.copy_to_gpu(len(req_ids))
            prepare_budgets(
                self.history,
                result,
                self.previous_budgets,
                metadata,
                positions,
                self.max_model_len,
                int(max(counts)),
                self.policy.base_top_k,
                histogram=self.stats.histogram if self.stats is not None else None,
            )
        else:
            self._prepare_cpu(result, positions, len(req_ids))
        return result

    def _prepare_cpu(self, result: torch.Tensor, positions: torch.Tensor, num_reqs: int) -> None:
        for row in range(num_reqs):
            slot, previous_row, span, start, count, prompt_len = map(int, self.metadata.np[row])
            pos = positions[start : start + count].long()
            if torch.any((pos < 0) | (pos >= self.max_model_len)):
                raise ValueError("Spec-K target position is outside the model context")
            values = self.history[slot, pos].clone()
            if span:
                assert self.previous_budgets is not None
                values[-span:] = self.previous_budgets[previous_row, :span]
            values[pos < prompt_len] = self.policy.base_top_k
            result[start : start + count] = values
            self.history[slot, pos] = values
            if self.stats is not None:
                self.stats.record_cpu(values, pos >= prompt_len)
