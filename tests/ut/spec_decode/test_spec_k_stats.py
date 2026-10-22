# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from vllm_ascend.spec_k import SpecKBudgetState, SpecKPolicy
from vllm_ascend.spec_k_config import SpecKConfig
from vllm_ascend.spec_k_stats import SpecKStats


@pytest.fixture(autouse=True)
def forbid_npu_initialization(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Statistics unit tests must not initialize NPU")

    monkeypatch.setattr(torch.npu, "_lazy_init", fail)


class CpuBuffer:
    def __init__(self, *shape, dtype):
        self.cpu = torch.empty(shape, dtype=dtype)
        self.np = self.cpu.numpy()


def make_stats(monkeypatch, interval=100):
    output = []
    stats = SpecKStats(torch.device("cpu"), 6, interval, 128, 0)
    monkeypatch.setattr(stats, "_emit", output.append)
    stats.start_session()
    return stats, output


def test_averages_are_weighted_by_verified_tokens_and_flush_at_idle(monkeypatch):
    config = SpecKConfig(enabled=True, ppl_thresholds=(128, 128, 128))
    state = SpecKBudgetState(SpecKPolicy(config, 6, torch.device("cpu")), 2, 64, 128, CpuBuffer, 100)
    output = []
    monkeypatch.setattr(state.stats, "_emit", output.append)
    # Only a is scheduled: b's proposed budgets must not enter the average.
    state.set_drafts(["a", "b"], torch.tensor([[3] * 7 + [6], [6] * 8], dtype=torch.int32))
    state.prepare(["a"], np.array([8]), {"a": [0] * 7}, [3], torch.arange(8), 16)
    state.stats.complete_step()
    state.prepare(["a"], np.array([8]), {"a": [0] * 7}, [3], torch.arange(3, 11), 16)
    state.stats.complete_step()
    state.release(["a"])
    assert len(output) == 1
    result = output[0]
    assert result["final"] and result["reason"] == "requests_drained"
    assert result["window"]["tokens"] == 13
    assert result["window"]["top_k_sum"] == 45
    assert result["window"]["avg_top_k"] == 45 / 13
    assert result["window"]["histogram"]["3"] == 11
    assert result["window"]["histogram"]["6"] == 2
    state.stats.close()
    assert len(output) == 1  # No misleading empty final report at shutdown.


def test_profile_average_excludes_delayed_steps_and_following_steps(monkeypatch):
    stats, output = make_stats(monkeypatch)
    # Two non-profile steps, then exactly ten active profile steps.
    for _ in range(2):
        stats.record_cpu(torch.tensor([6]), torch.tensor([True]))
        stats.complete_step()
    stats.set_profile_active(True)
    for _ in range(10):
        stats.record_cpu(torch.tensor([3] * 7 + [6]), torch.ones(8, dtype=torch.bool))
        stats.complete_step()
    stats.set_profile_active(False)
    stats.record_cpu(torch.tensor([6]), torch.tensor([True]))
    stats.complete_step()
    stats.flush("requests_drained", final=True)
    profile = [item for item in output if item["reason"] == "profile_end"][0]
    assert profile["window_steps"] == 10
    assert profile["profile_total"]["tokens"] == 80
    assert profile["profile_total"]["avg_top_k"] == 3.375
    assert output[-1]["session_total"]["tokens"] == 83
    assert output[-1]["session_total"]["avg_top_k"] == 288 / 83


def test_periodic_snapshots_preserve_total_and_reset_for_next_workload(monkeypatch):
    stats, output = make_stats(monkeypatch, interval=2)
    for k in [3, 6, 4]:
        stats.record_cpu(torch.tensor([k]), torch.tensor([True]))
        stats.complete_step()
    stats.flush("requests_drained", final=True)
    assert len(output) == 2
    assert output[-1]["session_total"]["avg_top_k"] == 13 / 3
    stats.start_session()
    stats.record_cpu(torch.tensor([6]), torch.tensor([True]))
    stats.complete_step()
    stats.close()
    assert output[-1]["session"] == 2
    assert output[-1]["session_total"]["tokens"] == 1
    assert output[-1]["session_total"]["avg_top_k"] == 6


@pytest.mark.parametrize("value", [-1, True, "100", 1.5])
def test_invalid_logging_intervals_rejected(value):
    with pytest.raises(ValueError):
        SpecKConfig(log_stats_interval=value)


def test_histogram_cannot_wrap_before_snapshot():
    with pytest.raises(ValueError, match="capacity"):
        SpecKStats(torch.device("cpu"), 6, 1 << 30, 8, 0)


def test_logging_can_be_disabled():
    assert SpecKConfig(log_stats_interval=0).log_stats_interval == 0


def test_async_snapshots_do_not_wait_on_model_thread_or_reuse_live_buffers(monkeypatch):
    stats, output = make_stats(monkeypatch)
    gate = threading.Event()
    producer = object()
    original_empty = torch.empty

    class CopyStream:
        def wait_stream(self, stream):
            assert stream is producer

    class Done:
        def record(self, stream):
            assert stream is stats.copy_stream

        def synchronize(self):
            assert gate.wait(timeout=5)

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    stats.copy_stream = CopyStream()
    stats.executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(torch.npu, "current_stream", lambda *args: producer)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.npu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.npu, "Event", Done)
    monkeypatch.setattr(torch, "empty", unpinned_empty)
    try:
        stats.record_cpu(torch.tensor([3]), torch.tensor([True]))
        stats.complete_step()
        stats.flush("first")
        assert not output  # Model thread returned while copy completion is held.
        stats.record_cpu(torch.tensor([6]), torch.tensor([True]))
        stats.complete_step()
        stats.flush("second")
        assert not output
    finally:
        gate.set()
        stats.close()
    assert output[0]["window"]["avg_top_k"] == 3
    assert output[1]["window"]["avg_top_k"] == 6
    assert output[-1]["session_total"]["avg_top_k"] == 4.5
