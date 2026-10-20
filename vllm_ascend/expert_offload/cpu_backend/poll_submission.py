
"""Submit graph CPU MoE from an ordered D2H sequence, without ACL callbacks.

The int32 sequence is copied after all input copies on the same stream.
Only supported on coherent x86 host memory; pinned slots remain alive for the
engine lifetime. A missed sequence aborts rather than using stale inputs.
"""
import atexit,ctypes,json,logging,os,threading,time
import torch
from . import async_completion
_LOG=logging.getLogger(__name__)
_WORKER=None
_LOCK=threading.Lock()

class PollWorker:
    def __init__(self, device):
        self.device=device
        self.completion=async_completion.worker_for(torch.device("npu",device))
        self.lib=ctypes.CDLL(os.environ['KT_MOE_POLL_LIBRARY'])
        self.lib.register_flag.argtypes=[ctypes.c_void_p]
        self.lib.register_flag.restype=ctypes.c_int
        self.lib.poll_next.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint32)]
        self.lib.poll_next.restype=ctypes.c_int
        self.channels=[]
        self.alive=True
        self.rows=[]
        trace_path=os.environ.get('KT_MOE_POLL_TRACE')
        self.path=trace_path.replace('%p',str(os.getpid())) if trace_path else None
        self.thread=threading.Thread(target=self.run,name='kt_input_poll',daemon=True)
        self.writer=threading.Thread(target=self.write_rows,name='kt_poll_log',daemon=True)
        self.thread.start()
        if self.path:self.writer.start()
        atexit.register(self.close)
    def close(self):
        self.alive=False;self.thread.join(timeout=3)
        if self.path:self.writer.join(timeout=3)
    def register(self, backend,hidden,stream,ready,hostflag):
        # Publish Python channel metadata before the native scan count grows.
        i=len(self.channels)
        self.channels.append((backend,hidden,int(stream.npu_stream),ready,hostflag))
        native=self.lib.register_flag(hostflag.data_ptr())
        if native!=i:raise RuntimeError('CPU poll registration mismatch')
    def write_rows(self):
        with open(self.path,'a',buffering=65536) as f:
            while self.alive or self.rows:
                time.sleep(.5)
                rows,self.rows=self.rows,[]
                f.writelines(json.dumps(row)+'\n' for row in rows);f.flush()
    @torch.inference_mode()
    def run(self):
        try:
            torch.npu.set_device(self.device)
            os.sched_setaffinity(0,{int(os.environ.get('KT_MOE_POLL_CPU','6'))})
            os.setpriority(os.PRIO_PROCESS,threading.get_native_id(),-10)
            completion=self.completion
            stamp=ctypes.c_uint64();seq=ctypes.c_uint32()
            while self.alive:
                k=self.lib.poll_next(ctypes.byref(stamp),ctypes.byref(seq))
                if k==-1:continue
                if k<0:raise RuntimeError('CPU input sequence skipped')
                dispatch=time.monotonic_ns()
                backend,hidden,handle,ready,_=self.channels[k]
                task=backend._launch_pinned_forward((hidden,handle))
                submitted=time.monotonic_ns()
                completion.jobs.put((backend,hidden,ready,task))
                if self.path:self.rows.append({'channel':k,'seq':seq.value,'layer':backend.config.layer_idx,'task_id':task,'ready_observed_ns':stamp.value,'dispatch_ns':dispatch,'submit_end_ns':submitted,'tid':threading.get_native_id()})
        except BaseException:
            _LOG.exception('Polling CPU MoE submission failed');os._exit(1)

def prepare(backend):
    global _WORKER
    # Initialize outside graph capture; captured graphs only increment and copy.
    device=torch.npu.current_device()
    backend._poll_sequence_device=torch.zeros((64,16),device=torch.device('npu',device),dtype=torch.int32)
    backend._poll_sequence_host=torch.zeros((64,16),device='cpu',dtype=torch.int32,pin_memory=True)
    backend._poll_sequence_slots=[(backend._poll_sequence_device[i,:1],backend._poll_sequence_host[i,:1]) for i in range(64)]
    backend._poll_sequence_used=0
    with _LOCK:
        if _WORKER is None:_WORKER=PollWorker(device)

def record_submission(backend,hidden,stream,ready):
    slot=backend._poll_sequence_used
    if slot>=64:raise RuntimeError('CPU input poll graph slot limit reached')
    backend._poll_sequence_used+=1
    deviceflag,hostflag=backend._poll_sequence_slots[slot]
    _WORKER.register(backend,hidden,stream,ready,hostflag)
    deviceflag.add_(1)
    hostflag.copy_(deviceflag,non_blocking=True)
