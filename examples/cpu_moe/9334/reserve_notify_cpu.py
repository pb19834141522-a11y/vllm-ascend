#!/usr/bin/env python3
"""Reversible reservation of CPU 40/168 for bound Ascend notification work."""
import os,json,time,argparse,re
from pathlib import Path
C=Path('/sys/fs/cgroup/cpuset')
S=Path('/mnt/share/y50063808/vllm-ascend-workspace/.vaws-local/wait-jitter-20261015/cpu-reservation.json')
H=C/'y50063808_housekeeping'; N=C/'y50063808_npu_notify'
RES={40,168}
def parse(s):
 out=set()
 for x in s.strip().split(','):
  if x:
   a,*b=map(int,x.split('-'));out.update(range(a,(b[0] if b else a)+1))
 return out
def fmt(s):return ','.join(map(str,sorted(s)))
def write(p,v):p.write_text(str(v))
def start(t):return Path(f'/proc/{t}/stat').read_text().rsplit(')',1)[1].split()[19]
def users():
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    status=(p/'status').read_text()
    if re.search(r'^Kthread:\s*0$',status,re.M) and not re.search(r'^State:\s*Z',status,re.M):yield p
   except (FileNotFoundError,ProcessLookupError):pass
def audit():
 bad=[];count=0;owned={}
 state=S.parent/'control-reservation.json'
 if state.exists():
  for x in (json.loads(state.read_text()).get('threads',[]) if json.loads(state.read_text()).get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip() else []):
   try:
    if start(x['tid'])==x['start']:owned[x['tid']]=x.get('reserved_cpu',168)
   except (FileNotFoundError,ProcessLookupError):pass
 for p in users():
  for t in (p/'task').iterdir():
   try:
    a=os.sched_getaffinity(int(t.name));count+=1
    if int(t.name) in owned and a=={owned[int(t.name)]}:continue
    if a&RES:bad.append({'tid':int(t.name),'allowed':fmt(a)})
   except (FileNotFoundError,ProcessLookupError):pass
 return {'threads':count,'ordinary_threads_using_reserved':bad}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['apply','verify','release']);args=ap.parse_args()
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if args.mode=='verify':print(json.dumps(audit()));return
 if args.mode=='release':
  s=json.loads(S.read_text());assert s['boot']==boot
  if N.exists():write(N/'cpuset.cpu_exclusive',0)
  for x in sorted(s['groups'],key=lambda x:x['path'].count('/')):
   p=Path(x['path'])
   if p.exists():write(p/'cpuset.cpus',x['cpus'])
  control=S.parent/'control-reservation.json'
  if control.exists():
   for x in json.loads(control.read_text()).get('threads',[]):
    try:
     if start(x['tid'])==x['start']:
      write(C/x['cgroup'].lstrip('/')/'tasks',x['tid']);os.sched_setaffinity(x['tid'],set(x['affinity']))
    except (FileNotFoundError,ProcessLookupError):pass
  if N.exists():N.rmdir()
  if H.exists():
   for pid in (H/'cgroup.procs').read_text().split():
    try:write(C/'cgroup.procs',pid)
    except ProcessLookupError:pass
   H.rmdir()
  for x in s['threads']:
   try:
    if start(x['tid'])==x['start']:os.sched_setaffinity(x['tid'],parse(x['cpus']))
   except (FileNotFoundError,ProcessLookupError):pass
  write(C/'cpuset.sched_load_balance',s['root_load_balance'])
  write(Path('/sys/devices/virtual/workqueue/cpumask'),s['workqueue_mask'])
  s['released_ns']=time.time_ns();S.write_text(json.dumps(s));print('RELEASED');return
 if S.exists():
  old=json.loads(S.read_text())
  if old['boot']==boot and not old.get('released_ns'):
   assert old.get('complete'),'Partial application: inspect state or release first'
   result=audit();assert not result['ordinary_threads_using_reserved'], 'Isolation escaped; inspect before continuing'
   print(json.dumps(result));return
 assert not H.exists() and not N.exists()
 allcpu=parse((C/'cpuset.cpus').read_text());allowed=allcpu-RES;assert len(allowed)==254
 groups=[]
 for p in C.rglob('cpuset.cpus'):
  cpus=p.read_text().strip()
  if cpus and p.parent != C:
   assert parse(cpus)-RES, str(p)+' would become empty'
   groups.append({'path':str(p.parent),'cpus':cpus})
 threads=[];rootpids=[]
 for p in users():
  try:
   cg=(p/'cgroup').read_text()
   if any(line.split(':',2)[1]=='cpuset' and line.split(':',2)[2]=='/' for line in cg.splitlines()):rootpids.append(int(p.name))
   for t in (p/'task').iterdir():
    try:
     tid=int(t.name);a=os.sched_getaffinity(tid);assert a-RES, f'{tid} has only reserved CPUs'
     threads.append({'tid':tid,'start':start(tid),'cpus':fmt(a)})
    except (FileNotFoundError,ProcessLookupError):pass
  except (FileNotFoundError,ProcessLookupError):pass
 wq=Path('/sys/devices/virtual/workqueue/cpumask')
 s={'boot':boot,'groups':groups,'threads':threads,'root_pids':rootpids,'root_load_balance':(C/'cpuset.sched_load_balance').read_text().strip(),'workqueue_mask':wq.read_text().strip(),'began_ns':time.time_ns(),'complete':False}
 S.write_text(json.dumps(s))
 for x in sorted(groups,key=lambda x:-x['path'].count('/')):write(Path(x['path'])/'cpuset.cpus',fmt(parse(x['cpus'])-RES))
 H.mkdir();write(H/'cpuset.mems',(C/'cpuset.mems').read_text());write(H/'cpuset.cpus',fmt(allowed));write(H/'cpuset.sched_load_balance',1)
 for pid in rootpids:
  try:write(H/'cgroup.procs',pid)
  except ProcessLookupError:pass
 N.mkdir();write(N/'cpuset.mems',(C/'cpuset.mems').read_text());write(N/'cpuset.cpus',fmt(RES));write(N/'cpuset.cpu_exclusive',1);write(N/'cpuset.sched_load_balance',0)
 write(C/'cpuset.sched_load_balance',0)
 mask=int(s['workqueue_mask'].replace(',',''),16)
 for c in RES:mask&=~(1<<c)
 hexmask=f'{mask:064x}';write(wq,','.join(hexmask[i:i+8] for i in range(0,64,8)))
 # Keep every original restricted worker affinity; cpuset updates may expand it.
 for x in threads:
  try:
   if start(x['tid'])==x['start']:os.sched_setaffinity(x['tid'],parse(x['cpus'])-RES)
  except (FileNotFoundError,ProcessLookupError):pass
 result=audit();s['audit']=result;s['complete']=not result['ordinary_threads_using_reserved'];s['finished_ns']=time.time_ns();S.write_text(json.dumps(s))
 assert s['complete'],result
 print(json.dumps({'state':str(S),'modified_groups':len(groups),'saved_threads':len(threads),'audit':result}))
if __name__=='__main__':main()
