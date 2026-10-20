#!/bin/bash
set -euo pipefail
WS=/mnt/share/y50063808/vllm-ascend-workspace
python3 "$WS/.vaws-local/wait-jitter-20261015/release_irq_policy.py"
python3 "$WS/cpu_moe/reserve_notify_cpu.py" release
python3 - <<'PYRESTORE'
from pathlib import Path
w=Path('/mnt/share/y50063808/vllm-ascend-workspace')
p=w/'cpu_moe/run_cpu_moe.sh'
s=p.read_text();line='python3 /mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/pin_npu_notify_irq.py --affinity-only || exit 1\n'
p.write_text(s.replace(line,''))
PYRESTORE
