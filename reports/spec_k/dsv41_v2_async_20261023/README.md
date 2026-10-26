# V4.1/A3 V2 异步执行修复归档

[完整报告](REPORT.md)记录2026-10-23已完成的服务、ShareGPT和profiling结果。
[根目录说明](../../../README.md)汇总算法适配与复现配置。

- `summary.json`、`baseline_summary.json`：384条、并发32、固定1024输出的端到端统计。
- `actual_topk.json`：实际target验证位置预算，包含bonus，排除prefill/padding。
- `profile_inspection.json`：8rank × 29个完整target-to-target周期。
- `outside_summary.json`与CSV：图外分段和CPU重叠；子项可能重叠，不能重复相加。
- `compiler_evidence/`：修改前后预算准备内核的编译产物，修复后不再生成software scatter lock。
- `historical_*_tests.txt`：同功能代码在2026-10-23的CPU、interpreter、NPU回归证据。
- `CHECKS.md`：本次归档提交的检查与限制。

原始大体积trace、权重、ShareGPT正文和逐token生成文本不入仓；绝对路径是采集机器上的追溯索引。
这里的报告和JSON保留原采集内容，提交SHA是当时HEAD加工作区patch，最终修复现已纳入本分支。

Baseline与动态组物理卡不同，接受率也不同，每组仅一轮；端到端观察不能归因于纯MoE加速，
也不能作为同卡重复后的稳定收益结论。持续满并发窗口包含补位prefill，不是纯decode平台期。
本归档不补跑也不发布新的精度结论。

CSV及编译IR仅规范LF行尾与行末空格，数值和编译指令未改变；前后摘要见`archive_normalization.json`。
