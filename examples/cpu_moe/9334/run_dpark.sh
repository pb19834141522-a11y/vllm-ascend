#!/bin/bash
# Dual weights: CPU GGUF; NPU persistent preconverted NZ buffers.
# Prefetch disabled: 105 native graph prefetch is not validated.
# Preserve vllm failures when output is piped through tee.
set -o pipefail

# 基本参数
CARDS=3
PORT=8150
MAX_SEQS=8
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
KT_BUILD="/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/kt_async_runtime"
# Match CPU activation math to NPU MXFP8 when experts migrate.
export KT_MOE_MATCH_NPU_MXFP8=1


export KT_MOE_PIN_WEIGHTS=1
# NUMA0: MoE physical cores32-63; service support threads stay outside them.
export KT_MOE_CORE_OFFSET=32
export KT_MOE_POLL_SUBMIT=1
export KT_MOE_POLL_CPU=6
export KT_MOE_COMPLETION_CPU=7
export KT_MOE_COMPLETION_RT_PRIORITY=1
export KT_MOE_POLL_LIBRARY=/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/kt_async_runtime/kt_kernel/libkt_ready_poll.so
unset KT_MOE_POLL_TRACE
unset KT_MOE_RT_PRIORITY  # Continuous FIFO triggered RT throttling; pending refinement.

if [ "${CPU_MOE_PHASE_TRACE:-1}" = "1" ]; then
    export KT_MOE_CPU_TRACE="$PROF_RAW_DIR/cpu_compute_%p.jsonl"
else
    unset KT_MOE_CPU_TRACE
fi
# Record only actual CPU compute spans (including L/E/C/T counters).
# Disable internal lifecycle, queue, callback, resource and worker timing.
unset KT_MOE_LIFECYCLE_TRACE
unset KT_MOE_LIFECYCLE_WORKERS
# Clock alignment runs only at profiler start/stop, outside per-task timing.
export CPU_MOE_CLOCK_CALIBRATION=1
unset CPU_MOE_ASYNC_RESULT CPU_MOE_ASYNC_VALIDATE_OUTPUTS

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

NDE_ARRAY="78,78,78,120,78,78,120,78,78,78,120,78,78,78,78,78,78,78,78,120,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,120,78,78,78,120,78,78,120,256"
NDE_JSON="[$NDE_ARRAY]"



SPECULATIVE_CONFIG=$(cat <<JSON
{
    "method": "dspark",
    "num_speculative_tokens": 7
}
JSON
)

COMPILATION_CONFIG=$(cat <<JSON
{
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [
        1,2,3,4,8,16,24,32,48,64
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
            "transfer_one_expert": true,
            "transfer_policy": "load_balance",
            "balance_max_transfers": 32,
            "balance_cpu_expert_us": 70.0,
            "balance_cpu_extra_route_us": 10.0,
            "balance_copy_us": 450.0,
            "balance_npu_route_us": 2.0,
            "balance_min_gain_us": 20.0,
            "transfer_scheduler": "npu",
            "transfer_copy_library": "/mnt/data/y50063808/dspark/vllm-ascend/vllm_ascend/expert_offload/lib/libnpu_expert_copy_dispatch.so",
            "weight_format": "gguf",
            "weight_path": "/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/DeepSeek-V4-Flash-0731-GGUF/dsv4_layer{layer_idx}_mxfp4.gguf",
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
numactl --physcpubind=8-28,30-31 --membind=0 vllm serve /mnt/data/DeepSeek-V4-Flash-0731 \
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
    --max-num-batched-tokens 150000 \
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
    --speculative-config "$SPECULATIVE_CONFIG" \
    --profiler-config "$PROFILER_CONFIG" \
    --shutdown-timeout 60 \
    --additional-config "$ADDITIONAL_CONFIG"
