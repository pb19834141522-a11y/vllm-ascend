#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
DEVICES="6,7,10,11,12,13,14,15"
TP=8
DP=1
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=32
MAX_BATCHED_TOKENS=2048
MEMORY_UTILIZATION=0.85
# 动态 Top-K [128,128,128]；固定7 draft，target图模式、draft eager。
SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true,"enable_adaptive_verification":false}'
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}'
PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$RUN_DIR/profile\",\"delay_iterations\":100,\"max_iterations\":30,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
python3 "$RUN_DIR/check_devices.py" > "$RUN_DIR/devices_before_start.json"
exec vllm serve "$MODEL_PATH" \
 --served-model-name "$MODEL_NAME" --host 127.0.0.1 --port "$PORT" \
 --tensor-parallel-size "$TP" --data-parallel-size "$DP" --enable-expert-parallel \
 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
 --max-num-batched-tokens "$MAX_BATCHED_TOKENS" --gpu-memory-utilization "$MEMORY_UTILIZATION" \
 --block-size 128 --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
 --safetensors-load-strategy lazy \
 --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
 --quantization ascend --engram-config '{"cpu_offload":true}' \
 --speculative-config "$SPECULATIVE_CONFIG" --compilation-config "$COMPILATION_CONFIG" \
 --async-scheduling --additional-config "$ADDITIONAL_CONFIG" --profiler-config "$PROFILER_CONFIG"
