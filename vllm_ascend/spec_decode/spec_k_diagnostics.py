# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from vllm.logger import logger


class SpecKEntropyDiagnostics:
    """Persist every draft-token entropy while collecting light summaries.

    One writer is created by TP rank zero in each DP replica. The JSONL file is
    deliberately lossless at the token level; threshold buckets can therefore
    be recomputed offline without rerunning model inference.
    """

    def __init__(
        self,
        output_dir: str,
        *,
        dp_rank: int,
        base_top_k: int,
        ppl_thresholds: Sequence[float],
        log_interval: int,
    ) -> None:
        directory = Path(output_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        timestamp_ns = time.time_ns()
        self.path = directory / (
            f"spec_k_entropy_dp{dp_rank}_pid{os.getpid()}_{timestamp_ns}.jsonl"
        )
        self._file = self.path.open("x", encoding="utf-8")
        self._dp_rank = dp_rank
        self._base_top_k = base_top_k
        self._ppl_thresholds = tuple(float(value) for value in ppl_thresholds)
        self._log_interval = log_interval
        self._step = 0
        self._token_count = 0
        self._selected_count = 0
        self._raw_entropy_sum = 0.0
        self._corrected_entropy_sum = 0.0
        self._entropy_delta_sum = 0.0
        self._selected_corrected_entropy_sum = 0.0
        self._markov_lowered_count = 0
        self._top_k_counts: Counter[int] = Counter()
        self._selected_top_k_counts: Counter[int] = Counter()
        self._next_log_count = log_interval
        self._closed = False
        self._write_json(
            {
                "record_type": "metadata",
                "schema_version": 2,
                "dp_rank": dp_rank,
                "pid": os.getpid(),
                "base_top_k": base_top_k,
                "ppl_thresholds": list(self._ppl_thresholds),
            }
        )
        self._file.flush()
        atexit.register(self.close)
        logger.info("Spec-K entropy diagnostics will write complete token data to %s", self.path)

    def _write_json(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self._file.write("\n")

    def record_step(
        self,
        *,
        req_ids: Sequence[str],
        draft_token_ids: Sequence[Sequence[int]],
        corrected_entropies: torch.Tensor,
        raw_entropies: torch.Tensor,
        top_ks: torch.Tensor,
        selected_lengths: Sequence[int],
    ) -> None:
        """Write all proposed tokens and mark those retained by dynamic SD."""
        if self._closed:
            raise RuntimeError("Spec-K entropy diagnostics writer is closed.")
        num_reqs = len(req_ids)
        if not (
            len(draft_token_ids) == num_reqs
            and len(selected_lengths) == num_reqs
            and corrected_entropies.ndim == 2
            and raw_entropies.ndim == 2
            and top_ks.ndim == 2
            and corrected_entropies.shape[0] >= num_reqs
            and raw_entropies.shape[0] >= num_reqs
            and top_ks.shape[0] >= num_reqs
        ):
            raise ValueError("Spec-K entropy diagnostics inputs have inconsistent batch shapes.")

        step = self._step
        self._step += 1
        for row, (req_id, token_ids, selected_length) in enumerate(
            zip(req_ids, draft_token_ids, selected_lengths)
        ):
            width = len(token_ids)
            if (
                corrected_entropies.shape[1] < width
                or raw_entropies.shape[1] < width
                or top_ks.shape[1] < width
            ):
                raise ValueError(
                    "Spec-K entropy diagnostics buffers are narrower than the draft sequence."
                )
            selected_length = max(0, min(int(selected_length), width))
            for position, token_id in enumerate(token_ids):
                raw_entropy = float(raw_entropies[row, position])
                corrected_entropy = float(corrected_entropies[row, position])
                entropy_delta = corrected_entropy - raw_entropy
                top_k = int(top_ks[row, position])
                selected = position < selected_length
                raw_ppl = math.exp(raw_entropy)
                corrected_ppl = math.exp(corrected_entropy)
                self._write_json(
                    {
                        "record_type": "token",
                        "dp_rank": self._dp_rank,
                        "step": step,
                        "request_id": req_id,
                        "draft_position": position,
                        "draft_width": width,
                        "selected_length": selected_length,
                        "selected_for_verification": selected,
                        "draft_token_id": int(token_id),
                        # Legacy aliases continue to mean the logits consumed
                        # by Spec-K, i.e. the post-Markov distribution.
                        "entropy": corrected_entropy,
                        "ppl": corrected_ppl,
                        "raw_entropy": raw_entropy,
                        "raw_ppl": raw_ppl,
                        "corrected_entropy": corrected_entropy,
                        "corrected_ppl": corrected_ppl,
                        "markov_entropy_delta": entropy_delta,
                        "markov_ppl_ratio": corrected_ppl / raw_ppl,
                        "expert_budget": top_k,
                    }
                )
                self._token_count += 1
                self._raw_entropy_sum += raw_entropy
                self._corrected_entropy_sum += corrected_entropy
                self._entropy_delta_sum += entropy_delta
                self._markov_lowered_count += int(entropy_delta < 0.0)
                self._top_k_counts[top_k] += 1
                if selected:
                    self._selected_count += 1
                    self._selected_corrected_entropy_sum += corrected_entropy
                    self._selected_top_k_counts[top_k] += 1

        if self._token_count >= self._next_log_count:
            # Python's buffered writer keeps per-token overhead manageable on
            # full datasets. Flush at summary boundaries so an interrupted run
            # still retains nearly all records.
            self._file.flush()
            self.log_summary()
            while self._next_log_count <= self._token_count:
                self._next_log_count += self._log_interval

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "proposed_tokens": self._token_count,
            "selected_tokens": self._selected_count,
            "mean_raw_entropy": (
                self._raw_entropy_sum / self._token_count
                if self._token_count
                else None
            ),
            "mean_corrected_entropy": (
                self._corrected_entropy_sum / self._token_count
                if self._token_count
                else None
            ),
            "mean_markov_entropy_delta": (
                self._entropy_delta_sum / self._token_count
                if self._token_count
                else None
            ),
            "markov_lowered_entropy_fraction": (
                self._markov_lowered_count / self._token_count
                if self._token_count
                else None
            ),
            "selected_mean_entropy": (
                self._selected_corrected_entropy_sum / self._selected_count
                if self._selected_count
                else None
            ),
            "expert_budget_counts": dict(sorted(self._top_k_counts.items())),
            "selected_expert_budget_counts": dict(
                sorted(self._selected_top_k_counts.items())
            ),
        }

    def log_summary(self) -> None:
        logger.info("Spec-K entropy diagnostics summary: %s", self.summary())

    def flush(self) -> None:
        """Expose buffered token records to an evaluator while service stays up."""
        if not self._closed:
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.log_summary()
        self.flush()
        self._file.close()
        self._closed = True
