#!/usr/bin/env python3
"""Pin this container's queue and graph callback after their lifecycle TIDs appear."""
import os,json,time,subprocess
from pathlib import Path
W=Path('/mnt/share/y50063808/vllm-ascend-workspace');S=W/'.vaws-local/wait-jitter-20261015/control-reservation.json';N=Path('/sys/fs/cgroup/cpuset/y50063808_npu_notify');began=time.time();offset=0;log=None;done={};saved=[]
for attempt in range(9000):
 rows=subprocess.check_output(['docker','top','cpu_dspark','-eo','pid,comm'],text=True).splitlines()[1:]
 engines=[int(x.split()[0]) for x in rows if 'VLLM::Engine' in x]
 if len(engines)!=1:time.sleep(.1);continue
 engine=engines[0];status=Path(f'/proc/{engine}/status').read_text();pid=int(next(x.split()[-1] for x in status.splitlines() if x.startswith('NSpid:')))
 if log is None:
  files=sorted({p.resolve() for p in (W/'cpu_moe/cpu_profile').glob(f'*/cpu_compute_{pid}.jsonl') if p.stat().st_mtime>=began-2})
  if len(files)!=1:time.sleep(.1);continue
  log=files[0]
 with log.open() as f:
  f.seek(offset)
  while True:
   pos=f.tell();line=f.readline()
   if not line:break
   if not line.endswith('\n'):f.seek(pos);break
   x=json.loads(line)
   if x.get('event')!='cpu_moe.lifecycle':continue
   phase=x.get('phase')
   if phase not in ('queue_take','callback_enter') or phase in done:continue
   target=168 if phase=='queue_take' else 40
   matches=[]
   for p in Path(f'/proc/{engine}/task').iterdir():
    try:
     ns=next(y.split()[-1] for y in (p/'status').read_text().splitlines() if y.startswith('NSpid:'))
     if int(ns)==x['tid']:matches.append(p)
    except FileNotFoundError:pass
   assert len(matches)==1,(phase,matches)
   p=matches[0];tid=int(p.name);cg=next(y.split(':',2)[2] for y in (p/'cgroup').read_text().splitlines() if y.split(':')[1]=='cpuset')
   saved.append({'tid':tid,'start':(p/'stat').read_text().rsplit(')',1)[1].split()[19],'affinity':sorted(os.sched_getaffinity(tid)),'cgroup':cg,'reserved_cpu':target,'phase':phase})
   S.write_text(json.dumps({'threads':saved,'began_wall':began,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},indent=2));(N/'tasks').write_text(str(tid));os.sched_setaffinity(tid,{target});done[phase]=tid;print('PINNED',phase,tid,target,flush=True)
  offset=f.tell()
 if len(done)==2:break
 time.sleep(.1)
else:raise RuntimeError('Timed out waiting for CPU MoE lifecycle threads')
print('CONTROL_PINNING_COMPLETE',flush=True)
