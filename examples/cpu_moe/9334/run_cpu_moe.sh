#!/bin/bash
# Native MXFP4 shared DRAM owner; x86 CPU borrows the same six buffers.
# Prefetch disabled: 105 native graph prefetch is not validated.
# Preserve vllm failures when output is piped through tee.
set -o pipefail

# Apply the host reservation with run_cpu_moe_host.sh before container startup.
python3 /mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/pin_npu_notify_irq.py --affinity-only || exit 1

# Control/callback threads inherit these NUMA-0 CPUs; kt-kernel's worker pool
# retains its explicit compute-core bindings. This is affinity, not exclusivity.
export CPU_MOE_CONTROL_CPUS="${CPU_MOE_CONTROL_CPUS:-32-39}"
taskset -pc "$CPU_MOE_CONTROL_CPUS" "$$" >/dev/null || exit 1

# 基本参数
CARDS=1
PORT=8150
MAX_SEQS=4
SERVED_NAME=deepseek_v4
GPU_MEM=0.95
MOE_DEBUG=false

export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

PROFILE_ROOT=/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/cpu_profile
RUN_STAMP=$(date +%Y%m%d-%H%M%S)
PROF_RAW_DIR="$PROFILE_ROOT/$RUN_STAMP"

mkdir -p "$PROF_RAW_DIR"
ln -sfn "$PROF_RAW_DIR" "$PROFILE_ROOT/latest"

LOG_DIR=/mnt/share/y50063808/vllm-ascend-workspace/logs
# Service output remains on the terminal; no automatic log file.

export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"
mkdir -p "$PROF_RAW_DIR"

# CANN 环境
if ! python3 -c "import acl" >/dev/null 2>&1; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "[env] source CANN"
fi

# CPU MoE
KT_BUILD="${KT_BUILD:-/mnt/share/y50063808/vllm-ascend-workspace/.vaws-local/native-port-20261014/package}"


export KT_MOE_PIN_WEIGHTS=1

# Detailed CPU scheduling diagnosis; disable lifecycle/workers after diagnosis.
export KT_MOE_CPU_TRACE="$PROF_RAW_DIR/cpu_compute_%p.jsonl"
export KT_MOE_LIFECYCLE_TRACE=1
export KT_MOE_LIFECYCLE_WORKERS=1
unset CPU_MOE_PHASE_TRACE

unset KT_MOE_TASK_TRACE KT_MOE_PAGE_TRACE KT_MOE_PHASE_TIMING
unset VLLM_ASCEND_CPU_MOE_TRACE VLLM_ASCEND_PROFILE_DIAG_LOG

# 代码与算子路径
export PYTHONPATH="/mnt/data/y50063808/dspark/vllm-ascend:/mnt/data/y50063808/dspark/vllm:$KT_BUILD:${PYTHONPATH:-}"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 超时设置
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

# NPU
export ASCEND_RT_VISIBLE_DEVICES=$CARDS

export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=0

export ASCEND_CUSTOM_OPP_PATH=/mnt/data/y50063808/dspark/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}

export LD_LIBRARY_PATH=/mnt/data/y50063808/dspark/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}

# Profiler
export VLLM_TORCH_PROFILER_DIR=$PROF_RAW_DIR
export VLLM_SERVER_DEV_MODE=1

# Expert offload
export VLLM_ASCEND_OFFLOAD_THRESHOLD=32

NDE_ARRAY="84"
NDE_JSON="[$NDE_ARRAY]"



COMPILATION_CONFIG=$(cat <<JSON
{
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [
        1,2,3,4,8,16,24,32
    ]
}
JSON
)

PROFILER_CONFIG=$(cat <<JSON
{
    "profiler": "torch",
    "torch_profiler_dir": "$PROF_RAW_DIR",
    "torch_profiler_with_stack": false,
    "ignore_frontend": true
}
JSON
)

ADDITIONAL_CONFIG=$(cat <<JSON
{
    "enable_cpu_binding": false,
    "decode_stats_enabled": false,
    "decode_stats_path": "$LOG_DIR/cpu_moe_${RUN_STAMP}_stats",
    "expert_offload_config": {
        "expert_offload": true,
        "expert_substitution_enabled": false,
        "enable_multi_card": false,
        "num_device_experts": $NDE_JSON,
        "num_device_layers": 2,
        "cache_policy_enabled": true,
        "expert_prefetch_enabled": false,
        "expert_prefetch_num": 1,
        "hot_expert_preload": true,
        "hot_experts_file": "/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/expert_rank_gsm8k.json",
        "moe_offload_debug": $MOE_DEBUG,
        "cpu_moe": {
            "enabled": true,
            "backend": "kt_kernel",
            "transfer_one_expert": false,
            "transfer_policy": "first_miss",
            "weight_format": "native_mxfp4",
            "weight_path": "/mnt/data/DeepSeek-V4-Flash-0731",
            "layers": "all",
            "max_num_tokens": 100,
            "cpuinfer_threads": 32,
            "threadpool_count": 1,
            "numa_nodes": [0]
        }
    }
}
JSON
)

# 启动服务
vllm serve /mnt/data/DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --no-async-scheduling \
    --seed 1024 \
    --served-model-name "$SERVED_NAME" \
    --max-num-seqs "$MAX_SEQS" \
    --max-model-len 15000 \
    --max-num-batched-tokens 128 \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEM" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --aggregate-engine-logging \
    --api-server-count 1 \
    --safetensors-load-strategy prefetch \
    --kv-cache-dtype fp8 \
    --compilation-config "$COMPILATION_CONFIG" \
    --profiler-config "$PROFILER_CONFIG" \
    --shutdown-timeout 60 \
    --additional-config "$ADDITIONAL_CONFIG"
