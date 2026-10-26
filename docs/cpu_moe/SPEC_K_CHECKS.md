# Spec-K 移植检查

目标是把算法移入作者的A5 CPU MoE分支；此机器没有对应A5环境。
本次没有安装这份Ascend源码到现有A3推理环境，没有重新编译CPU/NPU库，也没有跑A5模型服务。

## 源码核查

- 路由裁剪发生在`AscendRoutedExperts._select_experts`出口，早于`apply_offloaded_experts`、
  `maybe_prepare_hybrid_routes`及专家迁移规划，CPU与NPU读取相同的裁剪结果。
- 原后端`HybridExpertExecutor.prepare`在查`log2phy`前屏蔽负ID/零权重。
  KT `operators/common.hpp::should_skip_expert`在访问专家权重前排除负ID，
  `operators/llamafile/moe.hpp`同时覆盖单token、prefill和空CPU路由完成路径。
- 对TP1全驻留层，`init_eplb_config`返回空expert map；无效路由使用active range之外的ID，
  避免对expert map越界索引，不把裁剪任务发送给专家0。
- 对照vLLM固定提交确认`WorkerProfiler.is_running`和动态投机配置字段存在。
- V1真实forward和dummy/capture均提供同一预算buffer；draft context不提供target预算。
- V1每请求position由起点加连续arange生成；仅TP1/PCP1/DCP1启用连续History寻址。
- 请求预算按持久slot保留，处理未调度请求、重排、抢占和历史重算；短尾draft宽度按请求保存，
  验证被截短时仍从原预算最后一列取得bonus预算。
- 关闭Spec-K时不分配预算/直方图，不改变通信路径、CPU提交或原MoE路由数值。

## 可执行检查

仅新增`tests/ut/spec_decode/test_spec_k_cpu_moe.py`一份文件，并复用已有hybrid executor测试。

```bash
pytest --noconftest -q tests/ut/spec_decode/test_spec_k_cpu_moe.py \
  tests/ut/expert_offload/test_hybrid_executor.py
TRITON_INTERPRET=1 pytest --noconftest -q tests/ut/spec_decode/test_spec_k_cpu_moe.py
```

检查在现有A3容器的CPU路径执行，使用已安装的PyTorch和vLLM辅助模块，
不是对A5固定Python3.11环境或完整vLLM v0.30.0服务的集成验证。
`--noconftest`跳过上游UT入口的全机npu-smi探测；测试不调用NPU执行。
真实Triton源码在CPU解释器中核对熵、预算History、直方图及负路由哨兵，不等于A5编译/重放验证。

- CPU合约及原hybrid executor检查：13 passed，2个interpreter专用用例默认跳过。
- 开启Triton CPU解释执行：11 passed。
- 新增模块及所有修改Python文件：Ruff check通过；新增Python文件format检查通过。
- 启动脚本bash语法和两组JSON参数检查、文档相对链接、Python AST和git diff检查已执行。
- 完整`bash format.sh ci`在独立临时worktree尝试：安装markdownlint hook环境时达到120秒上限，
  退出码124，尚未获得完整hook结果；针对改动文件的Ruff和Markdown检查通过。

这组结果不证明A5可启动、NPU内核正确、输出精度或吞吐收益。A5首测应按照移植说明逐步进行。
