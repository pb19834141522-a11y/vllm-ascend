#!/bin/bash
set -euo pipefail
WS=/mnt/share/y50063808/vllm-ascend-workspace
if ss -H -ltn 'sport = :8150' | grep -q .; then
  echo '8150 is already running; no second service started.' >&2
  exit 1
fi
python3 "$WS/cpu_moe/reserve_notify_cpu.py" apply
python3 "$WS/cpu_moe/pin_npu_notify_irq.py" --install-startup-policy
nohup python3 "$WS/cpu_moe/pin_cpu_moe_control.py" > "$WS/.vaws-local/wait-jitter-20261015/control-pin-startup.log" 2>&1 < /dev/null &
exec docker exec cpu_dspark bash "$WS/cpu_moe/run_cpu_moe.sh"
