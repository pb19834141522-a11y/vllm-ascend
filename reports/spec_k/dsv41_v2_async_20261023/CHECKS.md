# 本次归档提交的检查（2026-10-26）

## 本次重新执行

- CPU合约：`pytest -q tests/ut/spec_decode/test_spec_k_fused.py tests/ut/spec_decode/test_spec_k_v2.py tests/ut/spec_decode/test_spec_k_stats.py`，53 passed。
- 真实Triton CPU解释执行：`TRITON_INTERPRET=1 pytest -q tests/ut/spec_decode/test_spec_k_triton_interpreter.py`，39 passed。
- 修改的6个Python文件：Ruff 0.14.0 check和format --check均通过。
- 根目录个人README、V2文档及新增报告Markdown：markdownlint-cli 0.45.0通过。
- 独立启动器：bash语法检查通过，Baseline/动态组 × profiler关/开共4组参数解析通过；该检查没有加载模型。
- Markdown相对链接、归档JSON、修改Python的AST、上游README原文备份一致性通过。
- `git diff --check`通过；提交文件常见凭证模式扫描见`publication_validation.json`。

## 历史硬件证据

同功能修复在2026-10-23完成6项A3 NPU回归，含统计开/关各160轮、无逐轮同步的双图交错。
8卡384条端到端与延迟采集记录见本目录报告。本次归档没有重跑NPU、重新测性能或继续精度评测。

## 全仓格式检查限制

已按AGENTS.md在独立临时worktree执行 `timeout 180 bash format.sh ci`，
完整检查卡在GitHub pre-commit hook环境初始化，无法完成全部hook，不能视为全仓CI通过。
原日志为`format_full_publish_20261026.txt`。上面的Ruff、Markdown和功能检查均单独执行。
检查工具仅装在隔离的`/tmp/p00948912-dsv41-publish-lint-20261026`，没有改变推理容器依赖。
