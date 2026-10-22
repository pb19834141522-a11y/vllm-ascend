# V2 Spec-K 迁移验证（2026-10-22）

[迁移说明与候选启动配置](../../../SPEC_K_V2.md)。

- `cpu_tests.txt`：466 passed、2 skipped，含请求重排、抢占历史、post-Markov 预算、target 捕获/执行的 buffer 作用域及统计边界。
- `interpreter_tests.txt`：46 passed，真实 Triton CPU 解释执行。
- `npu_validation.json`：A3 真实内核与 NPUGraph 重放、固定预算地址、padding 槽位及请求重排验证。
- `npu_regression_test.txt`：仓内新增单卡图重放回归，4 passed。

单卡测试使用健康空闲的物理 14 卡，已结束；未启动模型服务。
上述结果不构成完整 8 卡 V4.1 服务或端到端性能验证。

当前代码在上一个 `dev/dsv41-spec-k-fused-20261022` 归档基础上开发；
原始 V1 性能报告与启动配置保持在原报告目录，本次没有覆盖。
