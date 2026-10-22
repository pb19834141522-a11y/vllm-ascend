# 迁移验证状态

- 相关 CPU 合约与 V1/V2 平台/runner 回归：466 passed、2 skipped。
- 实际 Triton CPU interpreter：46 passed。
- A3 单卡内核及 NPUGraph 重放：4 组通过。
- 仓内新增单卡图重放 pytest：4 passed；与上述微验证覆盖相同机制，不计作新的整机测试。
- 修改 Python 文件的 Ruff 0.14.0 check / format 检查通过。
- git diff --cached --check 与 Python AST / 归档 JSON 解析通过。

本轮重新运行 bash format.sh ci，60 秒后退出 124，仍停在 markdownlint 的 Node 环境安装阶段。
日志见 format_full.txt；不宣称完整 hook 集合或全仓 lint 已通过。
工具安装在独立临时环境，推理依赖没有改动。

尚未运行完整 V4.1 W4A8 TP8 的 target + draft 服务捕获、384 条请求或性能对照。
当前结论是算法的代码接线、数值合约和预算内核图重放通过，不是整机收益结论。
