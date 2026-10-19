# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_ascend.ascend_config import SpecKConfig


SPEC_K_HISTORY_INITIAL_CAPACITY = 64


class SpecKPolicy:
    """Convert draft logits into per-token expert budgets."""

    def __init__(
        self,
        config: SpecKConfig,
        base_top_k: int,
        device: torch.device,
    ) -> None:
        if base_top_k <= 0:
            raise ValueError("Spec-K requires a MoE target with a positive top-k.")
        if len(config.ppl_thresholds) >= base_top_k:
            raise ValueError(
                "spec_k_config.ppl_thresholds must contain fewer values than "
                f"the model top-k ({base_top_k}) so at least one expert remains active."
            )
        if base_top_k > torch.iinfo(torch.uint8).max:
            raise ValueError("Spec-K supports model top-k values up to 255.")

        self.base_top_k = base_top_k
        self._apply_last_token = config.apply_last_token

        thresholds = config.ppl_thresholds + (0.0,) * (base_top_k - len(config.ppl_thresholds))
        self._log_perplexity_boundaries = torch.tensor(thresholds[::-1], dtype=torch.float32, device=device).log_()

    def top_ks_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        top_ks, _ = self.top_ks_and_entropy_from_logits(logits)
        return top_ks

    def top_ks_and_entropy_from_logits(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return expert budgets and the unrounded entropy behind each budget.

        Entropy is returned in natural-log units with shape ``[B, S]``. Keeping
        it separate from the integer budgets lets opt-in diagnostics persist
        the complete distribution and replay different PPL thresholds later.
        """
        if logits.ndim != 3:
            raise ValueError(f"Spec-K logits must have shape [B, S, V], got {logits.shape}.")

        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # xlogy defines 0 * log(0) as zero for reduced-vocabulary logits.
        entropy = -torch.xlogy(probabilities, probabilities).sum(dim=-1)
        token_top_ks = (entropy.unsqueeze(-1) >= self._log_perplexity_boundaries).sum(dim=-1)
        token_top_ks = token_top_ks.to(dtype=torch.int32)

        if self._apply_last_token:
            last_top_k = token_top_ks.to(torch.float32).mean(dim=1).to(torch.int32)
        else:
            last_top_k = torch.full(
                (logits.shape[0],),
                self.base_top_k,
                dtype=torch.int32,
                device=logits.device,
            )

        top_ks = torch.cat((token_top_ks, last_top_k[:, None]), dim=1).contiguous()
        return top_ks, entropy.contiguous()


@dataclass(frozen=True, slots=True)
class SpecKHistoryUpdate:
    new_output_length: int
    accepted_top_ks: torch.Tensor


@dataclass(slots=True)
class SpecKRequestState:
    base_top_k: int
    _history: np.ndarray = field(init=False, repr=False)
    _output_length: int = field(init=False, default=0)
    pending_draft_top_ks: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self._history = np.empty(0, dtype=np.uint8)

    @property
    def output_top_ks(self) -> torch.Tensor:
        return torch.from_numpy(self._history[:self._output_length])

    @output_top_ks.setter
    def output_top_ks(self, values: torch.Tensor) -> None:
        values_np = values.to(device="cpu", dtype=torch.uint8).numpy()
        self._reserve(len(values_np))
        self._history[:len(values_np)] = values_np
        self._output_length = len(values_np)

    @property
    def output_length(self) -> int:
        return self._output_length

    def _reserve(self, length: int) -> None:
        if length <= len(self._history):
            return
        # Grow geometrically instead of copying the entire history each step.
        history = np.empty(
            max(length, 2 * len(self._history), SPEC_K_HISTORY_INITIAL_CAPACITY),
            dtype=np.uint8,
        )
        history[:self._output_length] = self._history[:self._output_length]
        self._history = history

    def reconcile_output_length(self, output_length: int) -> None:
        if output_length < 0:
            raise ValueError("Spec-K output history length became negative.")
        self._reserve(output_length)
        if output_length > self._output_length:
            self._history[self._output_length:output_length] = self.base_top_k
        self._output_length = output_length

    def finalize_step(
        self,
        update: SpecKHistoryUpdate,
        sampled_token_top_k: torch.Tensor,
    ) -> None:
        self.finalize_step_values(
            update.accepted_top_ks.to(device="cpu").numpy(),
            int(sampled_token_top_k.item()),
            update.new_output_length,
        )

    def finalize_step_values(
        self,
        accepted_top_ks: np.ndarray,
        sampled_token_top_k: int,
        new_output_length: int | None = None,
    ) -> None:
        """Append completed host mirrors without per-request Torch operations."""
        num_new_tokens = len(accepted_top_ks) + 1
        if new_output_length is None:
            new_output_length = self._output_length + num_new_tokens
        previous_length = new_output_length - num_new_tokens
        if previous_length < 0:
            raise ValueError("Spec-K output history length became negative.")
        self.reconcile_output_length(previous_length)
        self._reserve(new_output_length)
        self._history[previous_length:new_output_length - 1] = accepted_top_ks
        self._history[new_output_length - 1] = sampled_token_top_k
        self._output_length = new_output_length

    def set_pending_draft_top_ks(self, top_ks: torch.Tensor) -> None:
        self.pending_draft_top_ks = top_ks.to(dtype=torch.uint8, device="cpu").clone()

    def top_ks_for_positions(
        self,
        positions: np.ndarray,
        num_prompt_tokens: int,
        pending_draft_start_position: int,
        use_pending_draft: bool,
    ) -> torch.Tensor:
        result = np.empty(len(positions), dtype=np.int32)
        self.write_top_ks_for_positions(
            result, positions, num_prompt_tokens,
            pending_draft_start_position, use_pending_draft,
        )
        return torch.from_numpy(result)

    def write_top_ks_for_positions(
        self,
        result: np.ndarray,
        positions: np.ndarray,
        num_prompt_tokens: int,
        pending_draft_start_position: int,
        use_pending_draft: bool,
    ) -> None:
        """Fill an existing host buffer, including historical recompute rows."""
        result.fill(self.base_top_k)
        output_indices = positions - num_prompt_tokens
        history_mask = (output_indices >= 0) & (output_indices < self._output_length)
        if history_mask.any():
            result[history_mask] = self._history[output_indices[history_mask]]

        pending = self.pending_draft_top_ks
        if use_pending_draft and pending is not None:
            pending_indices = positions - pending_draft_start_position
            pending_mask = (pending_indices >= 0) & (pending_indices < pending.shape[0])
            if pending_mask.any():
                result[pending_mask] = pending.numpy()[pending_indices[pending_mask]]
