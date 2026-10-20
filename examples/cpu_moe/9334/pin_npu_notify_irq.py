#!/usr/bin/env python3
"""Pin one NPU notification IRQ; preserve other devices and irqbalance settings."""
import argparse,json,os,re,shlex,socket,struct,subprocess,time
from pathlib import Path

WS=Path('/mnt/share/y50063808/vllm-ascend-workspace')
STATE=WS/'.vaws-local/wait-jitter-20261015/irq-policy.json'
POLICY_DIR=WS/'cpu_moe/irqbalance_policy.d'
CONFIG=Path('/etc/sysconfig/irqbalance')

def notify_irq(text,pci):
    matches=[]
    pattern=re.compile(r'IR-PCI-MSIX-'+re.escape(pci)+r'\s+2-edge\s+non-trans msg_chan_rx_msg_notify(?:\s|$)')
    for line in text.splitlines():
        if pattern.search(line):matches.append(int(line.split(':',1)[0].strip()))
    if len(matches)!=1:raise RuntimeError(f'Expected one notification vector for {pci}; found {matches}')
    return matches[0]

def banned_irqs(setup):
    if not re.match(r'^SLEEP \d+ ',setup):raise RuntimeError('Unrecognized irqbalance setup response')
    return set(map(int,re.findall(r'\bIRQ (\d+)\b',setup)))

def config_with_policy(text):
    lines=text.splitlines(keepends=True)
    hits=[i for i,line in enumerate(lines) if re.match(r'^\s*IRQBALANCE_ARGS=',line)]
    if len(hits)!=1:raise RuntimeError('Expected one active IRQBALANCE_ARGS; refusing to replace other configuration')
    i=hits[0];value=lines[i].strip().split('=',1)[1]
    parsed=shlex.split(value)
    if len(parsed)!=1:raise RuntimeError('Complex IRQBALANCE_ARGS quoting requires manual review')
    args=shlex.split(parsed[0]);wanted='--policyscript='+str(POLICY_DIR)
    for arg in args:
        if arg.startswith('--policyscript') or arg in ('-l',):
            if arg==wanted:return text
            raise RuntimeError('An existing irqbalance policy is configured; refusing to override it')
    args.append(wanted)
    lines[i]='IRQBALANCE_ARGS="'+' '.join(args)+'"\n'
    return ''.join(lines)

def exchange(path,message):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        s.settimeout(4);s.connect(str(path))
        credentials=struct.pack('3i',os.getpid(),os.getuid(),os.getgid())
        s.sendmsg([message.encode()],[(socket.SOL_SOCKET,socket.SCM_CREDENTIALS,credentials)])
        result=[]
        while True:
            data=s.recv(65536)
            if not data:break
            result.append(data)
    return b''.join(result).decode()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npu',type=int,default=1);ap.add_argument('--cpu',type=int,default=40)
    ap.add_argument('--affinity-only',action='store_true')
    ap.add_argument('--install-startup-policy',action='store_true')
    ap.add_argument('--verify',action='store_true')
    args=ap.parse_args()
    if os.geteuid()!=0:raise RuntimeError('Root is required for this device IRQ setting')
    board=subprocess.check_output(['npu-smi','info','-t','board','-i',str(args.npu)],text=True)
    m=re.search(r'PCIe Bus Info\s*:\s*(\S+)',board)
    if not m:raise RuntimeError('NPU PCI address could not be identified')
    pci=m[1];irq=notify_irq(Path('/proc/interrupts').read_text(),pci)
    node=int(Path(f'/sys/bus/pci/devices/{pci}/numa_node').read_text())
    if node>=0 and not Path(f'/sys/devices/system/cpu/cpu{args.cpu}/node{node}').exists():
        raise RuntimeError(f'CPU {args.cpu} is not local to NPU {args.npu} NUMA node {node}')
    affinity=Path(f'/proc/irq/{irq}/smp_affinity_list')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    if args.affinity_only:
        if not (state.get('boot_id')==boot and state.get('pci')==pci and state.get('irq')==irq
                and state.get('cpu')==args.cpu and irq in state.get('runtime_banned_irqs',[])):
            raise RuntimeError('Host IRQ policy has not been applied for this boot. Run run_cpu_moe_host.sh on the host first.')
        affinity.write_text(str(args.cpu))
        assert affinity.read_text().strip()==str(args.cpu)
        print(json.dumps({'irq':irq,'pci':pci,'cpu':args.cpu,'mode':'affinity-only'}));return
    if Path('/.dockerenv').exists():raise RuntimeError('Run full policy setup on the host, not inside the container')
    sockets=list(Path('/run/irqbalance').glob('irqbalance*.sock'))
    if len(sockets)!=1:raise RuntimeError(f'Expected one irqbalance daemon socket, found {len(sockets)}')
    sock=sockets[0];setup=exchange(sock,'setup');old_bans=banned_irqs(setup)
    if args.verify:
        assert irq in old_bans,setup
        assert affinity.read_text().strip()==str(args.cpu)
        print(json.dumps({'irq':irq,'pci':pci,'cpu':args.cpu,'verified':True,'setup':setup}));return
    STATE.parent.mkdir(parents=True,exist_ok=True)
    old_affinity=affinity.read_text().strip()
    if not state or state.get('boot_id')!=boot:
        state={'boot_id':boot,'pci':pci,'irq':irq,'original_affinity':old_affinity,
               'original_runtime_banned_irqs':sorted(old_bans),'original_setup':setup}
    STATE.write_text(json.dumps(state,indent=2))
    if args.install_startup_policy:
        before=CONFIG.read_text();after=config_with_policy(before)
        if 'original_config' not in state:
            backup=STATE.parent/'irqbalance.before.conf';backup.write_text(before);state['original_config']=str(backup)
        POLICY_DIR.mkdir(exist_ok=True)
        policy=POLICY_DIR/'y50063808_npu_notify.sh'
        script='''#!/bin/sh
# Only this NPU's message notification vector is owned by this policy.
case "$1" in
  */PCI_PLACEHOLDER)
    if awk -v irq="$2:" '$1 == irq && $0 ~ /IR-PCI-MSIX-PCI_PLACEHOLDER[[:space:]]+2-edge[[:space:]]+non-trans msg_chan_rx_msg_notify/ { found=1 } END { exit !found }' /proc/interrupts; then
      printf 'ban=true\\n'
      exit 0
    fi
    ;;
esac
exit 1
'''.replace('PCI_PLACEHOLDER',pci)
        policy.write_text(script);policy.chmod(0o755)
        subprocess.run(['sh','-n',str(policy)],check=True)
        assert CONFIG.read_text()==before,'irqbalance config changed during setup; aborting'
        if not re.search(r'^IRQBALANCE_BANNED_CPULIST=',after,re.M):after+='\nIRQBALANCE_BANNED_CPULIST=\"40,168\"\n'
        if after!=before:CONFIG.write_text(after)
        state['startup_policy']=str(policy)
        state['startup_config_args']=next(x for x in after.splitlines() if x.startswith('IRQBALANCE_ARGS='))
    merged=old_bans|{irq}
    message='settings ban irqs '+' '.join(map(str,sorted(merged)))
    if len(message)>490:raise RuntimeError('Too many existing IRQ bans for irqbalance socket protocol; refusing to truncate')
    if merged!=old_bans:exchange(sock,message)
    # This Huawei build's UI accepts a CPU list (not a hexadecimal mask).
    current_setup=exchange(sock,'setup')
    cpu_mask=int(re.search(r'BANNED\s+(\S+)',current_setup)[1].replace(',',''),16)
    owned={args.cpu,args.cpu+128}
    if 'original_runtime_cpu_ban_mask' not in state:state['original_runtime_cpu_ban_mask']=0
    cpu_mask |= sum(1 << c for c in owned)
    cpu_list=','.join(str(c) for c in range(256) if cpu_mask & (1 << c))
    exchange(sock,'settings cpus '+cpu_list)
    state['runtime_cpu_ban_list']=cpu_list
    verify=exchange(sock,'setup')
    assert banned_irqs(verify)==merged,verify
    affinity.write_text(str(args.cpu));assert affinity.read_text().strip()==str(args.cpu)
    state.update({'runtime_banned_irqs':sorted(merged),'setup_after':verify,'cpu':args.cpu,
                  'irqbalance_socket':str(sock),'applied_wall_ns':time.time_ns()})
    STATE.write_text(json.dumps(state,indent=2))
    print(json.dumps({'irq':irq,'pci':pci,'cpu':args.cpu,'banned_irqs':sorted(merged),
                      'state':str(STATE),'daemon_restarted':False}))
if __name__=='__main__':main()
