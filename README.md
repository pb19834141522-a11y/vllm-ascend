# DeepSeek V4.1 Flash：A3 上的动态 Top-K 与图模式优化

这是我在 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) 基础上维护的个人实验分支，
用于将 Spec-K 动态专家预算接入 DeepSeek V4.1 Flash 的 A3 W4A8 融合 MoE 路径，
并通过 Model Runner V2 和异步执行减少算法新增的图外开销。

- 个人仓：[pb19834141522-a11y/vllm-ascend](https://github.com/pb19834141522-a11y/vllm-ascend)
- 当前分支：`dev/dsv41-spec-k-v2`
- 最新修复及实测：[V2异步修复报告](reports/spec_k/dsv41_v2_async_20261023/REPORT.md)
- 适配说明：[融合MoE](SPEC_K_FUSED.md)、[Model Runner V2](SPEC_K_V2.md)
- 上游项目介绍与原有链接：[README_UPSTREAM.md](README_UPSTREAM.md)

## 我做了什么

### 1. 将动态专家预算接入 V4.1 的融合通信计算算子

复用原 Spec-K 的熵阈值策略，在 DSpark 完成 Markov bias 修正后计算全词表熵，
为下一轮 target verify 的每个生成位置给出专家预算。原生最大 Top-K 为6，
示例阈值为 `[128,128,128]`。

V4.1 的 `DispatchFFNCombineW4A8` 将分发、专家计算和合并放在一个融合算子中。
本分支保持这条融合路径，裁剪完成物理ID映射后的路由：
保留专家原有权重，丢弃路由改到保留桶ID `num_experts`，权重置零；
unpermute 在读取返回结果前跳过零权重路由。
V4.1 有384个路由专家，保留桶ID为384，避免把被裁剪工作转移到专家0。

这样可减少有效专家行的通信和计算工作，同时保留宽度6的张量形状以支持图重放。
固定排序、量化、元数据和同步仍有开销，耗时不保证按专家数同比下降。
融合算子的profiling收益不能再拆成独立的Dispatch、GMM和Combine收益。

### 2. 把整个算法接到 Model Runner V2，支持 draft 与 target 入图

复用上游已有的 DSpark 图捕获与重放能力，没有另造一套 draft 图运行时。
为其接入 post-Markov 熵归约、固定地址预算输出、请求槽位映射和 target 输入预算。
预算在 NPU 上随每次图重放更新；图捕获时使用完整Top-6不会把运行预算固化为6。

适配涵盖请求重排、padding、拒绝修正后的实际位置、历史重算、抢占和请求结束回收。
prefill、draft 专家和默认末尾验证位置保持完整Top-6；仅 target 生成阶段应用动态预算。

```text
DSpark draft图：7个候选 + post-Markov熵归约 → 固定地址专家预算
    → target输入准备：请求槽位/实际位置对齐、预算历史更新
    → target图：路由裁剪 → DispatchFFNCombineW4A8 → 原有验证/采样
    → 进入下一轮；实际target预算按窗口异步统计
```

`num_speculative_tokens=7` 指7个draft位置，完整验证通常为7+1个位置。
当7个位置均用3专家、末位用6专家时，实际平均为 `(7*3+6)/8=3.375`，而不是3。
真实值还受prefill交界、调度和预算分布影响，以日志的实际target验证统计为准。

### 3. 减少 CPU 下发与同步，保留异步调度重叠

- 熵、预算和预算历史更新在设备上完成，CPU只准备请求映射等元数据，
  不需要把logits或动态Top-K数值取回CPU后再决定执行。
- CPU元数据上传使用独立快照，避免异步调度下一轮改写仍被H2D读取的源缓冲。
- 实际Top-K直方图复用预算准备内核计数；在窗口边界快照，
  通过独立copy stream和后台线程等待、汇总及写日志，避免模型worker为统计同步等待。
- 复用已有V4.1设备端状态与图模式能力，保持CPU提交可与NPU执行重叠。
  不把上游已有的V4.1 CPU长度同步豁免算作本分支新发明。

仍存在真实的数据依赖和少量图外准备；draft入图不意味着整个投机周期成为一张图。

### 4. 修复双图异步执行时的预算准备停滞

旧预算History按离散位置写入时，后端生成了 `sync_block_lock/unlock` 软件锁路径。
在“draft图 → 图外预算准备 → target图”交错执行中，最小复现与真实服务都出现停滞，
逐轮全设备同步只能掩盖问题，并显著增加图外时间。

V2保证每个请求的输入位置连续。本分支据此保留每tile真实首位置，
用连续地址展开History访问，计算语义不变，编译产物不再生成上述软件锁。
通用位置路径仍保留，只有满足约束的V2调用启用该分支。

修复后移除了诊断用的逐轮全设备同步；新增无逐轮同步的160轮双图交错NPU回归，
核对预算、路由权重和统计直方图。底层锁缓冲区复用的完整机制尚未单独证明，
已验证的是该编译路径的消融与修复后的模型级结果。

### 5. 处理形状与测量口径

修复 `_prune_routes` 在不同运行输入行数下的编译假设，减少正式decode新增编译变体；
对V4.1元数据、预算buffer、图捕获和动态请求生命周期补充回归。
性能测量按每rank相邻target图起点取完整周期，再对8rank等权平均，
区分target图跨度、融合MoE累计、draft图和其余图外时间，不把跨流duration相加当总延迟。

## 本轮观察到的效果

以下为2026-10-22/23已保存的一轮结果，**不是同卡重复后的稳定收益结论**。
Baseline与动态组均为V2、固定7 draft、target与draft入图；权重和负载相同，物理卡组不同。

Profiling：跳过100个worker step，记录30个step；每rank取29个完整周期，8rank等权平均。

| 指标，ms/轮 | Baseline | 修复后动态组 |
|---|---:|---:|
| Target图跨度 | 58.261 | 55.956 |
| 其中40次融合MoE累计 | 24.013 | 20.171 |
| Target图外，含draft | 12.636 | 13.109 |
| 其中draft图跨度 | 6.516 | 6.972 |
| 完整target-to-target周期 | 70.897 | 69.066 |

旧的逐轮同步诊断动态组图外为31.652ms，修复后为13.109ms，恢复接近Baseline，
并非图外耗时全面优于Baseline。子项存在包含关系，不能重复相加。

端到端：ShareGPT相同384条、并发32、固定输出1024、`ignore_eos=true`，不带profiler。

| 指标 | Baseline | 修复后动态组 |
|---|---:|---:|
| 全程输出吞吐，token/s | 1280.49 | 1617.79 |
| 平均TPOT，ms | 22.541 | 18.569 |
| Draft token接受率 | 40.24% | 49.10% |
| 每轮输出长度，接受draft数+1 | 3.817 | 4.437 |
| 实际target verify平均Top-K | 6 | 3.420 |

全程吞吐本轮增加约26.3%，TPOT降低约17.6%；接受率也发生变化，
因此不能把全部端到端变化归因于专家计算减少。稳态硬件完整周期改善约2.6%。
持续满并发窗口包含补位prefill，不称为纯decode平台期。
图内融合MoE变快、完整周期变快、端到端变快及任务精度是不同验证项。
准确率工作暂不在本次归档中继续，尚无完整配对精度结论。

[报告及原始数值索引](reports/spec_k/dsv41_v2_async_20261023/README.md)保留测试日志、
编译前后证据、逐rank统计和资源审计。大体积trace、模型权重和数据集不入仓。

## 固定版本与安装

| 组件 | 本分支使用的版本 |
|---|---|
| 硬件 | Ascend A3，当前整机性能验证为8设备TP8/DP1/EP8 |
| 基础镜像 | `quay.io/ascend/vllm-ascend:deepseek-v4.1-flash-a3` |
| vLLM | `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`，本地无源码修改 |
| vLLM-Ascend上游基础 | `9ce79dfbdedba31e5f4490581cd9e7dd8fa9c06d` |
| 本分支历史 | `7fc3a57`融合适配 → `636236d`V2接线 → 本分支后续异步修复 |
| PyTorch / TorchNPU | `2.10.0+cpu` / `2.10.0.post4` |
| CANN | `9.1.0` |
| 模型 | `DeepSeek-V4.1-Flash-1017-w4a8-Ascend`，测试使用已准备的Engram host运行权重 |

使用已匹配的开发环境；保留现有依赖，按用户约定进行可编辑安装：

```bash
# vLLM固定到上表提交，在vllm目录中执行。
VLLM_TARGET_DEVICE=empty pip install -v -e . --no-deps --no-build-isolation

# 在本仓目录中执行；首次部署必须同时构建扩展和融合内核。
COMPILE_CUSTOM_KERNELS=1 pip install -v -e . --no-deps --no-build-isolation
```

C++/AscendC变更需要重新构建；后续仅Python/Triton修改通常重启服务即可。
`--no-deps`不保证任意镜像依赖都兼容，也不要把整数W4A8权重路径与原生FP4/FP8路径混用。
这里不要求重复下载或再次量化已准备好的可运行权重。

## 启动

[独立启动模板](start_dsv41_spec_k.sh)的模型路径、设备、端口、阈值和并发参数均放在顶部。
默认固定7 draft、V2、target/draft图模式、TP8/DP1/EP8、Engram DDR。
将 `SPEC_K_ENABLED=false` 可在同套参数下启动Baseline。
执行前自行确认所选8张卡健康空闲；脚本不重置设备，也不停止其他服务。

```bash
bash start_dsv41_spec_k.sh
```

关键配置：

```json
{
  "method": "dspark",
  "num_speculative_tokens": 7,
  "enforce_eager": false,
  "enable_adaptive_verification": false
}
```

Target配置为 `{"cudagraph_mode":"FULL_DECODE_ONLY"}`，
并设置 `VLLM_USE_V2_MODEL_RUNNER=1`、`enable_fused_mc2=1`和 `--async-scheduling`。
`enforce_eager=false`允许draft走图；target图模式独立配置。

可选profiler参数采用内置Torch-NPU Profiler的 `delay_iterations=100`、`max_iterations=30`，
不开启外部attach的 `PROFILING_MODE=dynamic`。先跑无profiler的端到端，再在同一服务触发采集。

## 支持边界与代码位置

当前适配保留V1，主要实测为V2 + DSpark + A3整数W4A8融合MoE。
配置对PP/PCP/DCP、动态验证长度、LoRA、EPLB及KV-transfer等不支持组合显式限制。
TP1/DP8/EP8等拓扑不能直接继承本次TP8性能结论，其他模型与后端需要单独验证。
Engram host offload与权重准备本身不是动态Top-K算法；其耗时也计入真实推理路径。

| 内容 | 主要文件 |
|---|---|
| 阈值配置与支持检查 | `vllm_ascend/spec_k_config.py`、`vllm_ascend/ascend_config.py` |
| 设备预算、请求槽位与历史 | `vllm_ascend/spec_k.py` |
| 熵归约、预算准备、路由裁剪 | `vllm_ascend/ops/triton/spec_decode/spec_k.py` |
| V2 draft图接线 | `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` |
| V2 target输入与图作用域 | `vllm_ascend/worker/v2/model_runner.py`、`vllm_ascend/ascend_forward_context.py` |
| 异步统计 | `vllm_ascend/spec_k_stats.py` |
| 融合MoE与路由 | `vllm_ascend/ops/fused_moe/`、`vllm_ascend/quantization/`、`csrc/mc2/dispatch_ffn_combine/` |
| CPU与解释执行回归 | `tests/ut/spec_decode/test_spec_k_*.py` |
| NPU双图异步回归 | `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_spec_k_v2_graph.py` |

工作基于vLLM、vLLM-Ascend、DSpark和此前Spec-K实现。个人贡献是V4.1/A3融合路径接入、
V2算法接线、异步执行修复和配套验证，并不把上游已有的draft图运行时或原算法归为个人首创。
保留上游版权声明与许可证，参见 [LICENSE](LICENSE)。
