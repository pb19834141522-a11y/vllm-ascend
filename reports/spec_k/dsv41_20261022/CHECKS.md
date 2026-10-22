# 提交前验证记录

## 已通过

- 修改涉及的 20 个 Python 文件：Ruff 0.14.0 `check` 与 `format --check` 通过。
- Ruff 修正了导入排序和空行并格式化 8 个文件；已检查改动，无推理算法变更。
- `git diff --cached --check` 通过；归档 CSV 统一 LF 行尾，统计数值未变。
- 格式化后 CPU 合约回归：250 passed，见 `cpu_after_format.txt`。
- 格式化后真实 Triton CPU interpreter：40 passed，见 `interpreter_after_format.txt`。
- 本轮 V2 DSpark / Engram 图合约：105 passed，见 `v2_graph_contract_tests.txt`。
- 权重转换工具回归：见 `conversion_tests.txt`。
- Python AST 与归档 JSON 解析通过；暂存文件的 GitHub token / 私钥常见模式检查未发现匹配。

## 全仓检查尚未完成

已按 AGENTS.md 运行 `bash format.sh ci`。
首次开发容器没有 pre-commit；在独立临时环境补齐检查工具后，容器内 GitHub hook 初始化长时间无进展。
随后在宿主可访问 GitHub 的独立工具环境重新运行完整命令，180 秒后退出 124，
停在 markdownlint 的 Node 环境安装阶段，尚未执行完整 hook 集合。
初始化日志见 `format_full_attempt.txt`。不能把这次尝试视为全仓 lint / Markdown / Gitleaks 已通过。
已单独完成上述 Ruff、diff 和功能检查；完整 CI 应在 hook 工具可用后补跑。

工具仅安装在 `/tmp/p00948912-spec-k-lint-*` 等独立目录，没有安装、卸载或升级推理环境依赖。
本轮未重新启动模型，也未将 V2 Spec-K 或 gating 裁剪融合实现为可运行功能。
归档的真实 NPU 内核、图重放与 8 卡服务结果来自本次保存前的同功能代码，详见完整报告。
