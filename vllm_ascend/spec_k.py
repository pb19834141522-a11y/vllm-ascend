# SPDX-License-Identifier: Apache-2.0

from collections import deque
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from vllm_ascend.spec_k_config import SpecKConfig
from vllm_ascend.spec_k_stats import SpecKStats

SPEC_K_METADATA_WIDTH = 7


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


class SpecKBudgetState:
    """Keep budgets/history on device; CPU builds only per-request metadata.

    History stores the budget actually used at each target input position.
    Recording rejected rows is safe: the next verification overwrites them;
    historical recompute only addresses committed positions. Slots survive
    preemption and grow outside target graphs when necessary.

    ``contiguous_positions`` is only valid when each request's positions are
    consecutive, as guaranteed by V1 input preparation for this TP1/PCP1 port. Other callers retain
    the generic position-gather path.
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
        contiguous_positions: bool = False,
    ):
        self.policy = policy
        self.max_model_len = max_model_len
        self.contiguous_positions = contiguous_positions
        device = policy.log_thresholds.device
        self.history = torch.full((max_num_reqs, max_model_len), policy.base_top_k, dtype=torch.int32, device=device)
        self.input_top_ks = torch.full((max_num_tokens,), policy.base_top_k, dtype=torch.int32, device=device)
        self.metadata = make_buffer(max_num_reqs, SPEC_K_METADATA_WIDTH, dtype=torch.int32)
        self.slots: dict[str, int] = {}
        self.free_slots = deque(range(max_num_reqs))
        self.previous_rows: dict[str, tuple[int, int]] = {}
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
            self.history = torch.cat((self.history, torch.full_like(self.history, self.policy.base_top_k)))
            if self.previous_budgets is not None:
                self.previous_budgets = torch.cat(
                    (self.previous_budgets, torch.full_like(self.previous_budgets, self.policy.base_top_k))
                )
            self.free_slots.extend(range(old_size, old_size * 2))
        slot = self.free_slots.popleft()
        self.history[slot].fill_(self.policy.base_top_k)
        self.slots[req_id] = slot
        return slot

    def set_drafts(self, req_ids: Sequence[str], budgets: torch.Tensor) -> None:
        if budgets.ndim != 2 or budgets.shape[0] != len(req_ids):
            raise ValueError("Spec-K draft budget rows do not match the proposer request batch")
        if self.previous_budgets is None:
            self.previous_budgets = torch.full(
                (self.history.shape[0], budgets.shape[1]),
                self.policy.base_top_k,
                dtype=torch.int32,
                device=budgets.device,
            )
        if budgets.shape[1] > self.previous_budgets.shape[1]:
            self.previous_budgets = torch.nn.functional.pad(
                self.previous_budgets,
                (0, budgets.shape[1] - self.previous_budgets.shape[1]),
                value=self.policy.base_top_k,
            )
        width = budgets.shape[1]
        budgets = torch.nn.functional.pad(
            budgets, (0, self.previous_budgets.shape[1] - width), value=self.policy.base_top_k
        )
        slots = [self._slot(req_id) for req_id in req_ids]
        indices = torch.tensor(slots, dtype=torch.int64).to(budgets.device, non_blocking=True)
        self.previous_budgets.index_copy_(0, indices, budgets)
        self.previous_rows.update((req_id, (slot, width)) for req_id, slot in zip(req_ids, slots))

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
        offset = 0
        for row, (req_id, count) in enumerate(zip(req_ids, counts)):
            previous_row, pending_width = self.previous_rows.get(req_id, (-1, 0))
            drafts = len(scheduled_drafts.get(req_id, ()))
            span = min(drafts + 1, int(count), pending_width) if drafts and previous_row >= 0 else 0
            self.metadata.np[row] = (
                self._slot(req_id),
                previous_row,
                span,
                offset,
                count,
                num_prompt_tokens[row],
                pending_width,
            )
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
                contiguous_positions=self.contiguous_positions,
            )
        else:
            self._prepare_cpu(result, positions, len(req_ids))
        return result

    def _prepare_cpu(self, result: torch.Tensor, positions: torch.Tensor, num_reqs: int) -> None:
        for row in range(num_reqs):
            slot, previous_row, span, start, count, prompt_len, pending_width = map(int, self.metadata.np[row])
            pos = positions[start : start + count].long()
            if torch.any((pos < 0) | (pos >= self.max_model_len)):
                raise ValueError("Spec-K target position is outside the model context")
            values = self.history[slot, pos].clone()
            if span:
                assert self.previous_budgets is not None
                values[-span:] = self.previous_budgets[previous_row, :span]
                # A shortened verification still ends with a bonus prediction.
                values[-1] = self.previous_budgets[previous_row, pending_width - 1]
            values[pos < prompt_len] = self.policy.base_top_k
            result[start : start + count] = values
            self.history[slot, pos] = values
            if self.stats is not None:
                self.stats.record_cpu(values, pos >= prompt_len)


@torch.library.custom_op("vllm::ascend_spec_k_prune_cpu_moe_routes", mutates_args=("weights", "ids"))
def prune_cpu_moe_routes(weights: torch.Tensor, ids: torch.Tensor, budgets: torch.Tensor) -> None:
    """Prune logical routes before residency planning; neither device owns -1."""
    if ids.ndim != 2 or weights.shape != ids.shape or budgets.shape != ids.shape[:1]:
        raise ValueError("Spec-K routing tensors must have matching token dimensions")
    if ids.device.type == "npu":
        from vllm_ascend.ops.triton.spec_decode.spec_k import prune_routes

        prune_routes(weights, ids, budgets, -1)
    else:
        drop = torch.arange(ids.shape[1], device=ids.device)[None, :] >= budgets[:, None]
        weights.masked_fill_(drop, 0)
        ids.masked_fill_(drop, -1)


@prune_cpu_moe_routes.register_fake
def _prune_cpu_moe_routes_fake(weights, ids, budgets):
    return None
