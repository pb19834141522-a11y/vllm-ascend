# V4.1 Spec-K 与 W4A8 融合 MoE

本次把原 V4 的动态专家预算迁入 V4.1 的
`DispatchFFNCombineW4A8` 路径。关闭功能时仍沿用原通信选择逻辑；
开启功能时也不切换到 AllGather 或分离的 Dispatch/GMM/Combine。

当前已完成源码集成、CPU / Triton / A3 NPU 验证与 384 条端到端及延迟采集对照。
最新结果和可比性限制见[完整实验报告](reports/spec_k/dsv41_20261022/FULL_REPORT.md)。
该对照不是同卡独占重复实验，不能将融合算子的变化进一步拆成独立通信收益。

## 启用配置

在原有 `additional-config` 中合入以下字段，保留其他启动参数：

```json
{
  "enable_fused_mc2": 1,
  "spec_k_config": {
    "enabled": true,
    "ppl_thresholds": [128, 128, 128],
    "full_top_k_layer_range": [0, 0],
    "apply_last_token": false,
    "log_stats_interval": 100
  }
}
```

`full_top_k_layer_range` 使用 Python slice 语义，指定继续使用完整 Top-K 的层；
`[0,0]` 表示没有保留层。阈值必须为正数并按非递增顺序排列。

本次启动模板设置 7 个 draft token，预算缓冲按实际投机长度生成，长度不写死：

```json
{
  "method": "dspark",
  "num_speculative_tokens": 7,
  "enforce_eager": true,
  "enable_adaptive_verification": false
}
```

目标 decode 仍使用 `FULL_DECODE_ONLY` 图模式，draft 保留 eager。
暂不支持同时开启动态验证长度。[V2 算法接线与验证边界](SPEC_K_V2.md)已单独说明；
以下历史记录针对 V1。当前迁移针对 A3、Runner V1/V2、整数 W4A8 target、
EP 2–32 卡及 TP/DP 并行；不支持 PP/PCP/DCP、LoRA、动态 EPLB、混合 shared placement
和 KV-transfer。其他模型、量化类型、并行方式需要另外适配和验证。

## 预算与路由流程

1. DSpark logits 完成 Markov bias 修正后，在设备上计算词表熵 `H`。
   保留原策略：`K = 原始K - 阈值个数 + count(H >= log(阈值))`。
   无效分布回退到完整 Top-K。
2. 预算留在设备上，根据请求 ID 对齐到下一轮 target 输入。CPU 只准备请求元数据，
   不读取 logits 或 Top-K 数值；历史预算按实际 target 位置保存，以支持拒绝覆盖和重算。
3. prefill 位置始终使用完整 Top-K；draft 专家不裁剪。
4. target 专家选择和物理专家 ID 映射完成后，保留路由的原权重，
   将超出预算的后缀 ID 改为融合内核的保留桶 `num_experts`，对应权重置零。
   V4.1 为 384 个专家，因此保留桶 ID 为 384，不使用专家 0 或 -1。
5. 保持物理路由宽度 6，便于目标图重放。预算随运行数据变化，
   不因图捕获时使用 Top-6 而把预算固定成 6。
6. 融合内核仅对实际专家桶收取激活、执行 GMM 并回传结果；
   补丁使 unpermute 在读取返回值之前跳过零权重路由，避免读到未计算的数据。

保留桶已用于原算子的 padding mask，host tiling 使用
`expertPerRank * worldSize + 1`，真实专家的通信/GMM 循环不包含最后这个桶。
裁剪可减少这些有效专家行的工作，但固定宽度的排序、部分量化、元数据交换和同步
仍然存在，时间不保证按 Top-K 比例下降。

`apply_last_token=false` 延续原策略：末尾验证位置仍用 Top-6。
若 7 个 draft 预算均为 3，则完整 8 个验证位置平均为 `(7*3+6)/8=3.375`。
改为 `true` 时，末尾预算取前面预算均值的整数部分；上述情况完整平均为 3。
阈值的实际效果仍取决于 V4.1 的 logits，不能仅由阈值断言全部位置恒为 3。

## 必须配套编译

此次修改包含 C++ 扩展与 W4A8 融合 AscendC 内核；不能只更新 Python 源码。
在现有开发容器中运行：

```bash
cd /home/p00948912/dsv41_dev_20261019
bash rebuild_spec_k_fused.sh
```

该脚本加载现有 CANN/ATB 环境并执行：

```bash
COMPILE_CUSTOM_KERNELS=1 pip install -v -e . --no-deps --no-build-isolation
```

不卸载或更新依赖。运行时检查扩展 ABI 标记，缺少新扩展会明确报错。
标记不能识别手工混用新扩展与旧 CANN 二进制的情况，必须成套构建与部署。

启动模板为开发目录下的 `start_server_int4_spec_k_profile.sh`，
固定 7 draft、TP8/DP1/EP8、并发上限 32，profiler 延迟 100 step 后采集 10 step。
先在脚本里确认可见设备空闲，再启动。
采集请求仍可使用原 ShareGPT 负载，需把客户端结果目录和服务端 `RUN_DIR` 对齐。

## 验证记录与后续对照

CPU 测试与相关回归合计 160 项通过。包括真实 Triton 源码的 CPU 解释执行、
完整词表熵、投机长度 1/5/7、路由裁剪、物理 ID 映射、请求重排、拒绝后的历史覆盖、
preemption、TP/SP/DP padding，以及 CPU Dynamo 单图读取变化的预算。
通过的验证进程结束时 NPU 仍未初始化；CPU Dynamo 检查不能替代 NPU 图捕获验证。

日志：开发目录 `runtime/spec_k_fused_20261020/cpu_validation.log`。

还使用真实本地模型配置执行了 `EngineArgs.create_engine_config()`：
确认 7 draft、Spec-K 阈值、Runner V1 与 `FULL_DECODE_ONLY` 均解析成功，
过程未初始化 NPU。日志为同目录的 `config_validation.log`。

空闲卡可用后，先验证启动、输出有效性和 NPU 图重放；再以相同编译产物及 7 draft 配置
采集三组：关闭 Spec-K、开启且阈值 `[1.0]`、开启且阈值 `[128,128,128]`。
三组都保持 `enable_fused_mc2=1`，使用不同输出目录。后续核查实际 Top-K、采信率、
target 融合算子耗时、图外 draft 耗时和端到端吞吐。

## 2026-10-20 硬件验证

用户重新安装后，新扩展 ABI 与融合算子构建产物检查通过。
设备 2–9 上 TP8/DP1/EP8、7 draft、阈值 `[128,128,128]` 服务成功启动，
目标图捕获和重放成功；真实 NPU 的熵计算、裁剪与预算传递小规模检查通过。
32 并发 ShareGPT 请求均完成 1024 token。

`delay=100 / active=10` 的全部 8 rank 已离线解析并校验：每 rank 10 次 target 图，
400 次融合 MoE，router 输入 `32×384`。rank 0 另有 400 次路由裁剪，
逐层位于 Gating 与融合 MoE 之间，同为 Stream 89；重放窗口没有逐层 CPU 裁剪调用。

完整结果为：`/home/p00948912/vllm_profile/dsv41_spec_k_fused_draft7_20261020_104421`。
其中 `rank0/ASCEND_PROFILER_OUTPUT/trace_view.json` 可直接打开；本次服务已停止。
实际平均 Top-K 还未统计，旧 baseline 为 5 draft，需统一为 7 后再做性能配对。

上述验证记录对应首次集成时点；后续统一 7 draft 的配对测试与实际 Top-K 已完成，
以[2026-10-22 报告](reports/spec_k/dsv41_20261022/FULL_REPORT.md)为准。

## 实际 Top-K 日志

新增统计从已准备好的 target 输入预算计数，排除 prompt 位置及图 padding；
包含末尾完整预算、之后被拒绝的验证位置，重复/重算的位置按实际执行次数计入。
它不是只对已采信输出 token 求均值，也不是对所有 proposer 预测预算直接求均值。
有 full_top_k_layer_range 时，日志是动态层的预算均值，全模型值另需按保留层数折算；
当前 [0,0] 配置没有此差异。

默认每 100 个实际 target step 输出一次，空闲前请求全部结束时输出本轮 workload 总计。
只在每个 DP engine 的 TP rank 0 上计数，避免 TP8 重复统计 8 份。
DP8 的全局平均应使用所有 DP 的 top_k_sum 总和除以 tokens 总和。

worker 根据 profiler.is_running 切分实际采集窗口，因此 delay=100 的等待阶段
不会进入 profile_total。自动停止或显式 stop_profile 时输出该窗口总计。
日志前缀为 Spec-K target_verify_stats，JSON 包含：

- window：本次窗口的 tokens、top_k_sum、avg_top_k 和各 K 的 histogram。
- session_total：本次连续 workload 到当前为止的加权累计值。
- profile_total：实际 profiling 窗口的累计值，仅 scope=profile 时提供。
- dp_rank、session、profile_id、window_steps、reason：用于区分实验及统计窗口。

计数融合进现有预算准备内核；直方图定期在 producer stream 上快照并清零，
通过辅助 stream 异步复制，后台线程等待完成后打印。没有逐轮 .item()/D2H 同步，
但设备计数和窗口快照仍有开销；log_stats_interval=0 可完全关闭统计。

本次仅修改 Python/Triton，已有融合算子无需重新编译安装，重启服务即可加载。
新统计及相关 CPU/解释执行回归日志保存在开发目录 runtime/spec_k_stats_20261020。
统计路径现已通过真实 8 卡服务验证；最新端到端实际 Top-K 为 3.4166，
profiling 窗口为 3.4488，详见上述完整报告。历史 trace 的未知均值不回填。
