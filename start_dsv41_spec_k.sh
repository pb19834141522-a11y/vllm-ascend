#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# 在匹配的A3开发容器内运行。先修改这里；不会自动选卡或停止任何服务。
# 使用已准备好的、含Engram host布局的运行权重；此默认值是本机实测路径。
MODEL_PATH="/tmp/p00948912-dsv41-1017-runtime-20261021T081310Z"
MODEL_NAME="dsv41-1017-flash-dspark"
DEVICES="0,1,8,9,10,11,12,13"
PORT=8000
TP=8
DP=1
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=32
MAX_BATCHED_TOKENS=2048
MEMORY_UTILIZATION=0.85
DRAFT_TOKENS=7
SPEC_K_ENABLED=true                  # false = Baseline，保留其他推理配置
PPL_THRESHOLDS='[128,128,128]'
PROFILE_ENABLED=false                # true = 启用/start_profile与/stop_profile
PROFILE_DIR="/home/p00948912/vllm_profile/dsv41_manual"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
unset PROFILING_MODE

SPECULATIVE_CONFIG="{\"method\":\"dspark\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"enforce_eager\":false,\"enable_adaptive_verification\":false}"
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
ADDITIONAL_CONFIG="{
  \"ascend_compilation_config\": {\"enable_npugraph_ex\":true,\"enable_static_kernel\":false},
  \"enable_cpu_binding\":true,
  \"enable_fused_mc2\":1,
  \"enable_flashcomm1\":true,
  \"enable_shared_expert_dp\":true,
  \"spec_k_config\": {
    \"enabled\":$SPEC_K_ENABLED,
    \"ppl_thresholds\":$PPL_THRESHOLDS,
    \"full_top_k_layer_range\":[0,0],
    \"apply_last_token\":false,
    \"log_stats_interval\":100
  }
}"

PROFILE_ARGS=()
if [ "$PROFILE_ENABLED" = true ]; then
  PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"delay_iterations\":100,\"max_iterations\":30,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
  PROFILE_ARGS=(--profiler-config "$PROFILER_CONFIG")
fi

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
  --async-scheduling --additional-config "$ADDITIONAL_CONFIG" "${PROFILE_ARGS[@]}"
