# Spec-K 的 Model Runner V2 迁移

本分支在已保存的 V4.1 融合 Spec-K 版本上增加 V2 接线，保留 V1。
复用已有 DSpark / DFlash ACL graph manager、捕获、重放和注意力参数更新流程。
不新增 draft 图运行时，不改原采样或拒绝采样算法，也不在这次迁移中做 gating 裁剪融合。

## 启动差异

在原来可运行的配置上选择 V2，并允许 draft 入图：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

```json
{
  "method": "dspark",
  "num_speculative_tokens": 7,
  "enforce_eager": false,
  "enable_adaptive_verification": false
}
```

Target 继续使用 `{"cudagraph_mode":"FULL_DECODE_ONLY"}`。
保持原来的 A3、整数 W4A8、EP、TP/DP 配置与 `enable_fused_mc2=1`；动态配置保持：

```json
{
  "spec_k_config": {
    "enabled": true,
    "ppl_thresholds": [128, 128, 128],
    "full_top_k_layer_range": [0, 0],
    "apply_last_token": false,
    "log_stats_interval": 100
  }
}
```

`speculative_config.enforce_eager=true` 仍然可用：V2 draft 保持 eager，Spec-K 语义一致，
便于与 draft 图模式配对。Target 的图模式独立配置。

本次只改 Python/Triton，已有整套 Spec-K C++/W4A8 内核的 editable 安装重启即可；
首次部署仍须按 [SPEC_K_FUSED.md](SPEC_K_FUSED.md) 构建对应扩展与融合内核。
配套 vLLM 固定为 `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`。
没有新增、卸载或升级推理依赖。

## 迁移的接口

1. `AscendDSparkSpeculator._sample_logits` 在上游已经加完 Markov bias 后记录熵归约的部分结果，
   采样仍委托上游实现；`_generate_draft` 完成后合并部分结果并输出预算。
2. 预算输出是固定设备 buffer，按 V2 的 `sample_idx_mapping` 写入持久请求槽位。
   padding 的 `-1` 槽位不写入，未调度请求的旧预算不被其他 batch 覆盖。
   输出写入位于已有 draft 图捕获范围内，重放不依赖 Python 重新生成预算属性。
3. `propose` 返回后只发布 CPU 已知的请求 ID/槽位对应，不读取预算数值，不引入 D2H。
4. V2 target 输入准备复用 `SpecKBudgetState`，读取 GPU 上拒绝修正后的实际 position，
   按原始 prompt 长度区分 prompt 与 generation，并更新预算历史。
5. runner 在执行与捕获的作用域内向平台 forward-context hook 提供同一个 target buffer。
   ContextVar 只保存这次调用的 tensor 引用，退出或异常后恢复，避免跨 runner 或 eager draft 泄漏。
6. 抢占清除待验证预算的槽位映射，保留位置历史；结束请求释放历史。
   idle/dummy 执行恢复完整 Top-6，内存探测和图捕获不计入正式统计。
7. 统计复用 worker 已有的 profiler 边界、异步快照和 shutdown 接口；每个 DP 的 TP0 计数。

保持原有语义：post-Markov 熵、保留权重不重新归一化、prompt 全 6、默认 bonus 全 6、
reserved bucket 为 `num_experts`、draft 不裁剪、保留层不裁剪、TP/SP 切分保持一致。
当全部 7 个 draft 预算为 3、bonus 为 6 时，实际 8 个 target 输入位置平均 Top-K 为 3.375。

V2 的 `dspark_draft_topk` 会截断词表分布，与当前全分布熵语义不兼容，启用 Spec-K 时明确拒绝。
原有 PP/PCP/DCP、动态验证、LoRA、EPLB、KV transfer 等限制仍保留，不因迁移放开。

## 开销与验证边界

为避免额外保存一整份 `[requests,7,vocab]` 修正 logits，NPU 只保存小的熵归约部分结果。
7 draft 对应 7 次 partial kernel 与 1 次合并，均可随已有 draft 图捕获。
这比 V1 的整块两段归约有更多设备 kernel，能否由较少的 CPU 下发抵消，需要实测。
迁移本身不保证图外等待消失或吞吐提高。

验证包括：

- CPU 接口回归 466 passed、2 skipped，包括 V1、V2、forward context 和平台路径。
- Triton CPU interpreter 46 passed，覆盖实际内核、请求映射、padding、不同投机长度和词表边界。
- A3 真实设备上的熵归约、预算槽位和 NPUGraph 重放验证，1/7 draft、bonus 开/关共 4 组通过。
- 仓内新增真实 NPU 回归 `test_spec_k_v2_graph.py`：4 passed。
- CPU 用上游真实 Markov 采样循环验证开启算法后 draft IDs 不变，预算来自修正后的 logits。

详细记录在开发目录 `runtime/spec_k_v2_20261022`，归档结果在
[reports/spec_k/dsv41_v2_20261022](reports/spec_k/dsv41_v2_20261022/README.md)。

**尚未验证完整 V4.1 W4A8、TP8、7 draft 服务的捕获/重放、384 条端到端和性能。**
单卡内核图重放不等价于整机 draft 模型全图；后续应先验证 V2 baseline，再比较 V2 Spec-K，
并统一设备、请求和图模式。原来的 V1 报告继续保留，不能用它代替 V2 性能结果。
