# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from vllm.logger import logger


def summarize_top_ks(histogram: list[int]) -> dict[str, Any]:
    tokens = sum(histogram[1:])
    top_k_sum = sum(k * count for k, count in enumerate(histogram))
    return {
        "tokens": tokens,
        "top_k_sum": top_k_sum,
        "avg_top_k": top_k_sum / tokens if tokens else None,
        "histogram": {str(k): count for k, count in enumerate(histogram) if k},
    }


class SpecKStats:
    """Aggregate actual target input budgets without a worker-thread D2H wait.

    Only TP rank zero of each DP engine owns this object. The existing budget
    preparation kernel updates its small int32 histogram. At window boundaries
    we snapshot/reset on the producer stream, copy on a separate stream, and
    wait/log on one background thread. CPU int totals cannot wrap across long
    server runs. Profiler transitions split windows at the actual delayed
    capture boundary, rather than at the /start_profile trigger.
    """

    def __init__(self, device: torch.device, base_top_k: int, interval: int, max_tokens: int, dp_rank: int):
        if interval <= 0 or interval * max_tokens > torch.iinfo(torch.int32).max:
            raise ValueError("Spec-K stats interval exceeds the int32 histogram capacity")
        self.device = device
        self.base_top_k = base_top_k
        self.interval = interval
        self.dp_rank = dp_rank
        self.histogram = torch.zeros(base_top_k + 1, dtype=torch.int32, device=device)
        self.steps = 0
        self.pending_steps = 0
        self.session = 0
        self.profile_active = False
        self.profile_id = 0
        self.session_open = False
        self._totals: dict[int, list[int]] = {}
        self._profile_totals: dict[tuple[int, int], list[int]] = {}
        self.copy_stream = torch.npu.Stream(device=device) if device.type == "npu" else None
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="spec-k-stats")
            if self.copy_stream is not None
            else None
        )

    def start_session(self) -> None:
        self.session += 1
        self.steps = 0
        self.session_open = True

    def record_cpu(self, budgets: torch.Tensor, valid: torch.Tensor) -> None:
        self.histogram.add_(torch.bincount(budgets[valid].long(), minlength=self.base_top_k + 1).to(torch.int32))

    def complete_step(self) -> None:
        self.steps += 1
        self.pending_steps += 1
        if self.pending_steps >= self.interval:
            self.flush("interval")

    def set_profile_active(self, active: bool) -> None:
        if active == self.profile_active:
            return
        self.flush("profile_end" if self.profile_active else "before_profile")
        self.profile_active = active
        if active:
            self.profile_id += 1

    def flush(self, reason: str, final: bool = False) -> None:
        if not self.pending_steps and (not final or not self.session_open):
            return
        metadata = {
            "dp_rank": self.dp_rank,
            "tp_rank": 0,
            "session": self.session,
            "scope": "profile" if self.profile_active else "workload",
            "profile_id": self.profile_id if self.profile_active else None,
            "step_end": self.steps,
            "window_steps": self.pending_steps,
            "reason": reason,
            "final": final,
        }
        self.pending_steps = 0
        if final:
            self.session_open = False
        snapshot = self.histogram.clone()
        self.histogram.zero_()
        if self.copy_stream is None:
            self._consume(snapshot.tolist(), metadata)
            return
        cpu = torch.empty(self.base_top_k + 1, dtype=torch.int32, device="cpu", pin_memory=True)
        producer = torch.npu.current_stream(self.device)
        with torch.npu.stream(self.copy_stream):
            self.copy_stream.wait_stream(producer)
            cpu.copy_(snapshot, non_blocking=True)
            done = torch.npu.Event()
            done.record(self.copy_stream)
        assert self.executor is not None
        self.executor.submit(self._consume_async, snapshot, cpu, done, metadata)

    def _consume_async(self, device_snapshot, cpu, done, metadata) -> None:
        try:
            # This wait runs off the model worker thread. Keep both buffers
            # alive until the copy completes, even if the request has ended.
            with torch.npu.device(self.device):
                done.synchronize()
            self._consume(cpu.tolist(), metadata)
            del device_snapshot
        except Exception:
            logger.exception("Failed to read asynchronous Spec-K statistics")

    def _consume(self, histogram: list[int], metadata: dict[str, Any]) -> None:
        session = metadata["session"]
        total = self._totals.setdefault(session, [0] * len(histogram))
        for k, count in enumerate(histogram):
            total[k] += count
        result = {**metadata, "window": summarize_top_ks(histogram), "session_total": summarize_top_ks(total)}
        if metadata["scope"] == "profile":
            key = session, metadata["profile_id"]
            profile = self._profile_totals.setdefault(key, [0] * len(histogram))
            for k, count in enumerate(histogram):
                profile[k] += count
            result["profile_total"] = summarize_top_ks(profile)
            if metadata["reason"] == "profile_end" or metadata["final"]:
                self._profile_totals.pop(key, None)
        self._emit(result)
        if metadata["final"]:
            self._totals.pop(session, None)

    @staticmethod
    def _emit(result: dict[str, Any]) -> None:
        logger.info("Spec-K target_verify_stats %s", json.dumps(result, separators=(",", ":")))

    def close(self) -> None:
        self.flush("shutdown", final=True)
        if self.executor is not None:
            self.executor.shutdown(wait=True)
