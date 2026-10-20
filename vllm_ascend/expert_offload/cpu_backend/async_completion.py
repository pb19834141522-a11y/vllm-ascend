"""CPU completion signals readiness for direct mapped-host result consumption."""
from __future__ import annotations
import atexit
import ctypes
import logging
import os
import queue
import threading
import torch

_LOG = logging.getLogger(__name__)
_WORKERS = {}
_LOCK = threading.Lock()
_ACL = None

class CompletionWorker:
    def __init__(self, device):
        import torch_npu
        self.device = device
        self.stream = torch_npu.npu.SyncLaunchStream(device=device)
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self.run, name="kt_cpu_h2d", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def close(self):
        self.jobs.put(None)
        self.thread.join(timeout=5)

    @torch.inference_mode()
    def run(self):
        torch.npu.set_device(self.device)
        # This thread blocks on jobs/CPU completion; it is not a busy poller.
        cpu = os.environ.get("KT_MOE_COMPLETION_CPU")
        priority = int(os.environ.get("KT_MOE_COMPLETION_RT_PRIORITY", "0"))
        if cpu is not None:
            os.sched_setaffinity(0, {int(cpu)})
        if priority:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        from kt_kernel import kt_kernel_ext as ext
        trace = bool(os.environ.get("KT_MOE_LIFECYCLE_TRACE"))
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                backend, hidden, ready, task_id = job
                if trace:
                    ext.lifecycle_set_context(task_id, backend.config.layer_idx)
                    ext.lifecycle_mark("completion_wait_enter")
                backend._wrapper.drain_pinned_forward()
                if trace:
                    ext.lifecycle_mark("completion_wait_end")
                # CPU writes are complete. The NPU combine kernel reads the
                # mapped host result directly after this readiness event.
                ready.record(self.stream)
                if trace:
                    # Use the phase name accepted by the existing C++ trace binding.
                    ext.lifecycle_mark("output_ready_recorded")
        except BaseException:
            _LOG.exception("CPU MoE completion failed; terminating this engine instead of releasing stale output")
            os._exit(1)

def worker_for(device):
    index = device.index if device.index is not None else torch.npu.current_device()
    with _LOCK:
        if index not in _WORKERS:
            _WORKERS[index] = CompletionWorker(index)
        return _WORKERS[index]

def record_submission(backend, hidden, stream, ready):
    global _ACL
    worker = worker_for(hidden.device)
    if _ACL is None:
        _ACL = ctypes.CDLL("libascendcl.so")
        _ACL.aclrtLaunchCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        _ACL.aclrtLaunchCallback.restype = ctypes.c_int
    args = (hidden, int(stream.npu_stream))
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def submit(_unused):
        try:
            task_id = backend._launch_pinned_forward(args)
            if os.environ.get("KT_MOE_LIFECYCLE_TRACE"):
                from kt_kernel import kt_kernel_ext
                kt_kernel_ext.lifecycle_mark("callback_exit")
            worker.jobs.put((backend, hidden, ready, task_id))
        except BaseException:
            _LOG.exception("CPU MoE submission failed")
            os._exit(1)
    backend._async_callback_refs.append(submit)
    from kt_kernel import kt_kernel_ext
    kt_kernel_ext.launch_no_block_callback(
        ctypes.cast(submit, ctypes.c_void_p).value, 0, args[1])
