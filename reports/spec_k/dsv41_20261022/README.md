# V4.1 Spec-K 代码与实验归档（2026-10-22）

- [完整报告及算子明细](FULL_REPORT.md)
- [Draft 图模式与新增算子优化核查](OPTIMIZATION_REVIEW.md)
- [逐 rank 对比](rank_comparison.csv)
- [Target 算子](target_operators.csv)
- [图外默认流算子](outside_default_stream_operators.csv)
- [图外事件等待](outside_event_wait_by_order.csv)
- [完整统计](details.json)
- [功能与构建说明](../../../SPEC_K_FUSED.md)

## 固定版本

- vLLM-Ascend 基础提交：`9ce79dfbdedba31e5f4490581cd9e7dd8fa9c06d`。
- 配套 vLLM：`c4df37dfcc7f72ca41cd43801a468526ecf99a2b`；该仓工作区无改动。
- 本分支包含整数 W4A8 融合路由裁剪、设备预算与异步统计、V4.1 元数据优化、
  `_prune_routes` 运行时行数修复、FP4/FP8 转整数权重工具及对应测试。
- 当前 Spec-K 运行在 Model Runner V1；target 使用图模式，DSpark draft 使用 eager。

## 数据口径与复查

无 profiler 的端到端测试为 ShareGPT 384 条、并发 32、固定输出 1024、ignore EOS。
同一服务随后以 32 条长请求触发 profiler，跳过 100 step，记录 30 step。
硬件统计取每 rank 的相邻 target 图起点形成的 29 个完整周期，再对 8 rank 等权平均。
Target 计入主流的完整执行跨度，包括裁剪内核与末尾事件；图外为该图结束到下图开始。
算子 duration、跨流工作与等待可能重叠，不应跨流直接相加视为关键路径。

Baseline 与动态组使用不同物理卡组，且监测到外部上下文；各配置只有一轮数据。
本报告描述本轮观察，不替代同卡独占重复实验，也没有评估裁剪后的任务准确率。
“持续满并发窗口”仍包含补入请求的 prefill，不是纯 decode 平台期。

两份 `*_config.json` 与 `*_start_server.sh` 是实际采集配置快照。
启动脚本依赖原实验目录的 `environment.sh` 和 `check_devices.py`，不是独立启动器。
大体积原始 trace、模型权重、ShareGPT 数据与生成文本未入仓；完整报告末尾列出宿主机路径。

## 已完成验证

保存的验证日志来自本次提交前相同功能代码：CPU 合约测试 250 passed，
Triton interpreter 40 passed，A3 内核与图重放校验通过，真实 NPU 行数回归测试通过。
`npu_validation.json` 保留多种行数共用同一编译变体的结果。
真实 8 卡服务已完成两组端到端与 30 step 采集；正式动态请求期间未新增裁剪编译缓存变体。

本轮补充的 V2 / Engram 图合约测试 105 passed；适配结论见 `OPTIMIZATION_REVIEW.md`。
格式检查结果见 `CHECKS.md`。
