# DSpark 入图与 Spec-K 开销核查

核查日期：2026-10-22。本文是源码适配分析，不是新优化后的性能结果。
本轮没有修改推理算法或切换运行模式；代码归档保持已验证的 V1 target 图 / draft eager 路径。

## 1. 结论与实测边界

1. **已有可复用的 V2 DSpark draft 全图实现。** 捕获范围不仅是模型前向，也包含顺序 Markov 采样；当前 V1 proposer 则明确禁用 draft 图。
2. **本地 Spec-K 尚未接入 V2。** 配置检查直接拒绝 V2，预算生成与 target 输入准备目前挂在 V1。仅去掉检查或把 `enforce_eager` 改成 false 不构成适配。
3. **draft 入图是降低 CPU 提交与小算子开销的候选方案，不保证消除图外等待。** 数据依赖、模型计算、跨卡通信、Engram host 查询及下一轮输入准备仍然存在。
4. **`_prune_routes` 正是算法的路由裁剪实现。** 已在 target 图中，每图 40 次合计 0.9681 ms；有机会融合到现有 gating 的输出阶段，避免独立启动。
5. **图外新增 Spec-K 内核合计约 0.4792 ms。** 它们不能单独解释图外增加 2.1309 ms；增长主要表现为最后 Argmax 后的等待，仍需关联等待事件的生产者。

下表来自本目录[完整报告](FULL_REPORT.md)，单位为 ms/完整投机周期，8 rank × 29 周期等权平均。

| 指标 | Baseline | 动态 Top-K | 动态减 Baseline |
| --- | ---: | ---: | ---: |
| Target 图跨度 | 54.1744 | 51.2522 | -2.9222 |
| 融合 MoE 算子合计 | 24.6751 | 20.5430 | -4.1321 |
| 裁剪内核合计 | 0 | 0.9681 | +0.9681 |
| Target 图外跨度 | 43.2165 | 45.3474 | +2.1309 |
| 完整周期 | 97.3909 | 96.5996 | -0.7912 |
| 图外默认流内核合计 | 10.1681 | 10.5909 | +0.4227 |
| 图外默认流 EVENT_WAIT 合计 | 4.4285 | 6.0350 | +1.6065 |
| 最后 Argmax 结束至下一 target 开始 | 35.2605 | 37.4493 | +2.1888 |

这些跨度不等于某个单独子模型的运行时间。默认流未覆盖的约 28 ms 可能有辅助流、CPU 准备和排队，不能直接命名为 CPU 空闲或 Engram。
两组物理卡组不同、有外部上下文记录，且只有一轮；小幅差异仍需同卡重复实验确认。

## 2. Draft 图：现有代码到底支持到哪里

### 2.1 已存在的执行链

当前配套 vLLM 固定在 `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`。

| 层次 | 本地源码与关键函数 | 已有行为 |
| --- | --- | --- |
| V1 | `vllm_ascend/spec_decode/dspark_proposer.py`，`AscendDSparkProposer.__init__` | 强制 `self.use_cuda_graph = False` |
| V2 Ascend proposer | `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`，`init_cudagraph_manager` | 根据 speculative `enforce_eager` 决定模式；绑定 graph manager 和更新流 |
| V2 图运行时 | `vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py`，`DFlashAclGraphManager` | 捕获、重放及注意力参数更新；支持既有 ACL / updatable graph 分支 |
| V2 运行时注册 | `vllm_ascend/patch/worker/patch_v2/patch_dflash_speculator.py` | 将 Ascend graph manager 接入上游 speculator |
| vLLM DSpark | `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`，`_generate_draft` | 同一捕获函数执行 `_run_model` 后执行 `_sample_sequential` |
| vLLM draft 图 | `vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py`，`capture` | 将 context KV 预计算和 draft 函数一起捕获 |
| vLLM draft 重放 | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`，`propose` | FULL 分支调用 `query_cudagraph_manager.run_fullgraph`；未覆盖的 prefill context 仍在图外处理 |
| V4.1 模型 | `vllm_ascend/models/deepseek_v41/dspark.py` | 已有 V4.1 DSpark、SWA cache 和注意力适配 |
| 同权重草稿量化 | `vllm_ascend/patch/worker/patch_v2/patch_dspark.py` | 同 checkpoint draft 继承 target 整数量化配置，避免缺少 scale / offset 参数 |
| V4.1 状态 | `vllm_ascend/models/deepseek_v41/engram/model_state.py` | 已有 V2 Engram、图 padding 和 replay 元数据路径 |

`speculative_config.enforce_eager=true` 在 V2 也会禁用 draft 图；target 的图模式独立控制。
迁移候选条件是 V2 加 draft `enforce_eager=false`，固定 7 draft、自适应验证关闭，target 图覆盖原有 batch。
这只是适配后的候选配置，不能直接用于尚未接入 V2 的本分支 Spec-K。

### 2.2 上游证据与限制

- [PR #15269：DSV4 DSpark full graph](https://github.com/vllm-project/vllm-ascend/pull/15269) 已合入。
  这是实际适配工作，不是仅提出的 RFC；但其正文中那组测试 full graph 为 779.95 token/s、eager 为 961.91 token/s，不能当作图一定更快的证据。
  该结果也不是我们的 V4.1 / W4A8 / 7 draft / TP8 配置。
- [PR #18750：V4.1 V2 FULL padding 修复](https://github.com/vllm-project/vllm-ascend/pull/18750) 已合入。
  其正文报告 A3、V4.1 W8A8、DP2 × TP8、DSpark、FULL_DECODE_ONLY 的真实权重验证。
  本地 `EngramModelState.prepare_attn` 已有对应的 padded replay_start 与 inactive replay 防护。
- 本地基础提交本身是 [PR #18831](https://github.com/vllm-project/vllm-ascend/pull/18831)：增加 A3 V4.1 MRV2 PCP nightly 配置。
  **但该 nightly decoder 的 draft 仍设置 `enforce_eager=true`。** 不能用它证明 V4.1 draft 全图已验证。
- 本地 V4.1 教程的一些段落仍写 A3 只支持 V1，同时源码、测试和合入 PR 已含 V2。
  因此应以明确的代码路径及目标配置实测为依据，不能只引用该教程作否定或肯定结论。

已有代码足以作为迁移起点；本轮没有完成 V4.1 W4A8、TP8、7 draft 全图的真实服务验证。
尤其需要验证 SWA/DSA-CP 图元数据、图 padding、Engram 重放和 TP/EP 通信模式一致性。

### 2.3 Spec-K 迁移不能漏的部分

1. **预算来源：** 当前在 V1 `llm_base_proposer.py` 的 Markov 循环结束后读取已原地修正的 logits。
   上游 V2 使用 `logits_i = base_logits[:, i] + bias`，不会改写 `base_logits`。
   若直接在循环末尾对 base_logits 算熵，会变成未加 Markov bias 的另一个算法。
   应在 V2 正确保留各位置的修正 logits，或融合计算其熵，再得到与 V1 一致的预算。
2. **输出地址：** 捕获函数的 Python 只在捕获时执行，重放时不会重新发布 Python 属性。
   预算应写入持久设备输出 buffer；runner 使用真实请求数与请求状态索引读取，不能只依赖捕获时的 `take_last_spec_k_budgets()`。
3. **请求与位置映射：** 接入 V2 request state、idx_mapping、拒绝后的实际 position、prefill/重算/完成/抢占和 padding。
   保持 prompt 全 6、bonus 全 6、按真实执行验证位置统计，避免历史预算错配。
4. **目标入口：** 在 V2 target 输入准备和捕获/重放入口传入稳定地址的 `token_top_ks`；保持 TP / sequence parallel 的预算切分与实际路由一致。
5. **公共路由：** 复用现有 `routed_experts` + 裁剪 + reserved bucket + unpermute 防护，draft 不裁剪，目标保留层不裁剪。
6. **统计：** 移植实际 delayed profiler 边界、TP0 统计、异步 histogram 快照。不能在 draft 捕获内部调用 Python 日志/线程提交。
7. **分布裁剪分支：** V2 `_sample_sequential_topk` 会截断词表分布。
   不能对截断后的分布直接计算原算法的全词表熵；应明确禁用这个组合或实现正确的全分布预算。
8. **放开配置：** 最后才删除 `_validate_spec_k` 对 V2 的拒绝，并加入启动、输出/预算一致性、图重放及端到端对照验证。

## 3. `_prune_routes` 与算法逐步对应

调用链：`routed_experts` 完成专家选择/物理 ID 映射 → `shard_token_budgets` →
`prune_fused_routes` custom op → Triton `_prune_routes` → `DispatchFFNCombineW4A8`。

以预算 3、目标 Top-6 为例：

```text
输入专家 ID: [5, 37, 80, 121, 206, 310]
输入预算:    3
输出专家 ID: [5, 37, 80, 384, 384, 384]
输出权重:    [w0, w1, w2, 0, 0, 0]
```

384 是当前模型融合内核的保留桶，未改成真实专家，也没有把专家发给其他 rank 计算。
保留路由的权重不重新归一化。图形状保持 `[N,6]`，实际专家工作行数下降。
预算按词表熵计算的部分在图外；这里仅执行预算，不重复计算熵、不做 CPU Top-K 计算。

该内核每次仅对后缀 ID 和权重写入，40 次合计 0.9681 ms，约 24.2 μs/次，
占动态 target 图 1.89%、完整周期 1.00%。这些是 duration 占比，不是可保证回收的端到端加速。
当前 TP8、每轮 256 target token 对应每 rank 32 路由行；32 × 6 = 192 个元素，
只启动一个 256 元素 Triton program，因此更应优先减少启动，而非期待简单调整 block 大幅提速。
该判断仍需独立 NPU 微基准与整体图测量确认。

### 可行的融合位置

仓内有可修改的 gating C++ / AscendC 实现：

- `vllm_ascend/ops/fused_moe/router/fused_topk_router.py` 调用 `moe_gating_top_k_hash`。
- `csrc/torch_binding.cpp` 提供 op schema / 调用封装。
- `csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_def.cpp` 定义算子输入。
- `csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_without_group.h` 的 `CopyOut(row)` 写出 weights / IDs。

**优先候选：** 为 gating 增加可选的设备预算输入，在已有选专家与权重缩放之后、输出之前直接写入保留桶和零权重。
这可能消除每层独立裁剪启动及额外全局内存写回，但需同步改 schema、meta、host tiling、kernel 和相关分支并重新编译。
必须覆盖 hash / 普通选择、保留层、draft、padding 和物理 ID 映射顺序；不能仅在一个 CopyOut 中修改便声称完成全部支持。
当前不支持的 EPLB 等组合不能因融合而被默默放开。

次选是在融合 MoE 的 init-routing 中直接消费预算，避免 gating ABI 改动；但融合内核接口与路由排序也需要改动，工作量更大。
单纯再包一层 Python custom op 不会自动将 opaque CANN 算子与 Triton 内核变成一次设备启动。

另有图内 MEMCPY_ASYNC 11 → 91 次、总时长 0.0088 → 0.0583 ms。
80 次增量恰好是每层两次，值得核对 mutation / functionalization / buffer alias 的编译结果。
**这里只观察到数量对应，尚未证明复制由哪个编译节点插入，不能直接删除。** 总 duration 增量约 0.0495 ms，优先级低于 40 次裁剪。

## 4. 图外新增算子的优化空间

| 内核 | 每周期调用 | 平均时长 ms | 代码职责 | 候选优化 |
| --- | ---: | ---: | --- | --- |
| `_entropy_partials` | 1 | 0.3152 | 全词表分块 max / exp sum / entropy moment | 调优归约；与 Markov 修正或采样融合，避免重复读 logits |
| `_entropy_budgets` | 1 | 0.0360 | 合并分块、比较 log 阈值、生成含 bonus 的预算 | 与分块后处理融合；persistent 输出；纳入 draft graph |
| `_prepare_budgets` | 1 | 0.1280 | 对齐请求与 position、更新历史和 target 预算、统计 | 接入 V2 设备 request mapping；融合 full-6 初始化/padding 与元数据操作 |
| 合计 | 3 | 0.4792 | — | 占图外 1.06%、完整周期 0.50% |

这些内核已经在 NPU 上计算，没有必须等 CPU 算完熵/预算数值的 D2H 往返。
CPU 在 `SpecKBudgetState.prepare` 中仍构造少量请求元数据，之后 H2D；它必须先提交，
target 才能读取对应预算，但不等于 CPU 读取和计算 NPU logits。

优化顺序建议：

1. 将 V2 完整 draft 路径接通，并把预算输出也纳入 graph，减少 Markov 小算子和预算算子的 Python 提交。
   图模式减少提交开销，不消除真实 Markov 数据依赖，也不消除熵归约本身的计算。
2. 让预算与 V2 持久 request state 共用设备映射，复用输出和临时 buffer。
   对 `fill_(6)` + `_prepare_budgets` 评估融合，但必须继续保证 padding、无草稿和重算位置安全。
3. 评估 entropy / argmax 共享 logits 读取的融合内核；保持 FP32 归约、阈值边界、无效分布回退及 post-Markov 语义。
   不用“argmax 差不多”或截断词表分布替代全词表熵。
4. 统计已有 TP0、每 100 step 异步快照设计；先测开/关统计的差异，不能未经测量将其认作主要瓶颈。

真正需要继续定位的是新增约 1.61 ms 的图外等待，尤其最后 Argmax 后的第 25/26 次 EVENT_WAIT。
需由 event ID、生产者 stream、HCCL / H2D / Engram 调用及 CPU launch flow 逐个关联，
区分小内核使后续提交推迟、已有通信暴露、Engram producer 延迟和运行波动。
当前不能直接把它们都命名为通信，或保证 draft 图会消除它们。

## 5. 后续验证顺序

1. 在隔离实验分支确认 V2 baseline 的 draft FULL 捕获/重放、固定 7 draft、W4A8、TP8、DSA-CP、Engram host 与当前模型可运行。
2. 再接入 Spec-K，核对逐位置预算、有效目标 Top-K、采信统计和 padding / 拒绝 / 请求重排的正确性。
3. 同卡配对比较 V2 draft eager 与 graph，baseline 和动态组都跑；这样能分离公共加速和 Spec-K 新增开销。
4. 保持同服务先跑无 profiler 的 384 条负载，再采集 delay100 / 30 step；沿用本报告 8 rank × 29 完整周期口径。
5. 单独评估 gating 裁剪融合，避免把 V2、算法阈值和融合内核变动同时合在一个性能结论中。

本轮额外运行 DSpark V2 graph contract、DSpark speculator 与 V4.1 Engram state 合约测试，105 passed。
这是 CPU mock 单元验证，不是上述整机图模式验证；日志见 `v2_graph_contract_tests.txt`。

图模式支持、功能正确、有效专家数减少、单周期变快、最终 TPOT/吞吐改善是五项不同证据，应分别验收。
