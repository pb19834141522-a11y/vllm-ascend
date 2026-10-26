# A5 CPU/NPU 混合 MoE：动态 Top-K 移植

本分支基于 yxk-fdu 的 `share/cpu-moe-9334-20261020`，加入我的 Spec-K 动态专家预算。
它不是把 A3 的 W4A8 二进制搬到 A5，也不升级原 CPU MoE 运行栈。
**目前只做了源码迁移、CPU合约检查和Triton CPU解释执行，尚未在A5启动或验证精度/性能。**

## 固定配套版本

| 组件 | 固定提交 |
|---|---|
| 本分支基础 vLLM-Ascend | `7137c79c1a523159b366ce4a687676811269776e` |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，`v0.30.0` |
| ktransformers | `7d38786e561d5a763f4f68f600676d93465234c7`，相同分享分支 |
| 算法来源 | 个人V4/V4.1 Spec-K；V4.1参考提交 `f700d705c97503da55faad246d86b49f204966a9` |

配套的CPU MoE、权重、构建和NUMA说明继续使用[原作者指南](README.zh.md)。
KT保留原版本：其C++计算前已跳过负ID，空路由也有零结果完成路径，因此本次无需修改KT。
本分支不携带权重、GGUF或编译产物。A5/x86_64环境必须自行编译，不能使用A3/aarch64/Python3.12的库。

## 迁移内容

1. 从DSpark逐位置Markov bias修正后的**完整词表logits**计算熵，沿用原PPL阈值到专家预算的规则。
2. 预算留在NPU，按请求槽位缓存；target使用拒绝修正后的真实position读取预算，保存历史以支持重算。
   请求重排、暂时未调度、抢占、结束和slot复用均有处理。CPU只构建小型请求元数据，并上传独立快照。
3. 在target专家选择之后、logical/physical映射及专家迁移/CPU任务提交之前，
   将超过预算的后缀专家ID改为`-1`并置零权重。保留路由顺序和原权重，不重新归一化。
4. CPU/NPU都只接收保留的路由，权重迁移规划同样不再看见丢弃的专家。
   原有CPU任务提交、NPU计算、等待及合并时序保持；即便没有CPU路由，仍由原后端完成任务协议。
5. 保持物理路由宽度和target预算buffer地址不变，图捕获/启动探测填充完整Top-K，重放读取实时预算。
   复用连续位置History寻址，避免前一版A3移植中遇到的软件scatter锁路径。
6. 实际target预算直方图在设备上累计，独立copy stream和后台线程输出统计；不为算法新增逐轮全设备同步。
   内置profiler按真实采集边界分窗。

对未offload的全驻留层，丢弃路由在NPU侧改成active expert range之外的保留值，
不会映射成专家0。本分支没有引入A3融合W4A8的保留桶或扩展ABI检查。

本次保持作者的**V1 runner**。CPU MoE路径原本即要求V1；DSpark draft沿用该路径的eager执行，
只保留原来的target图模式。这与V4.1分支的V2双图方案不同，不要开启`VLLM_USE_V2_MODEL_RUNNER=1`。

## 最小使用步骤

在A5已有可运行的原作者环境里切换这份Ascend代码；保持配套vLLM和KT版本。
若尚未准备原环境，先按原作者指南完成基线部署，不要直接用本机A3镜像。

```bash
git clone --branch dev/dsv4-a5-cpu-moe-spec-k \
  https://github.com/pb19834141522-a11y/vllm-ascend.git
cd vllm-ascend
python -m pip install -v -e . --no-deps --no-build-isolation
```

迁移本身只改变Python/Triton，不新增C++/AscendC修改；原作者的NPU搬运、合并和CPU内核仍须已构建。
第一次部署按照原作者文档完成整套构建。`--no-deps`保留配套依赖，不保证任意环境兼容。
确认`PYTHONPATH`、`vllm_ascend.__file__`指向这份代码。

现有启动脚本只需在`additional_config`加入：

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

其他CPU MoE配置保留；关闭额外的expert pruning、expert substitution和activation routing，
以免混用不同裁剪策略。保留`method=dspark`、`num_speculative_tokens=7`，并关闭动态投机验证长度。
本次移植明确支持原作者单卡 **TP1/PP1/DP1、无PCP/DCP**；其他组合在启动时明确拒绝。
LoRA、KV-transfer、EPLB、混合shared placement、静态内核和非默认compact drafter batch未适配。

[9334示例脚本](../../examples/cpu_moe/9334/run_dpark.sh)已加入开关，默认关闭：

```bash
# 先按你的A5机器修改脚本里的权重、源码、库路径、卡号和CPU/NUMA绑定。
SPEC_K_ENABLED=false bash examples/cpu_moe/9334/run_dpark.sh
SPEC_K_ENABLED=true SPEC_K_THRESHOLDS='[128,128,128]' \
  bash examples/cpu_moe/9334/run_dpark.sh
```

示例保留9334的机器路径，不是无需修改的通用启动器；脚本默认的CPU迁移策略及容量保持不变。
不必重新量化已经准备好的原生MXFP4/GGUF权重。

## A5 首轮检查

先确认关闭Spec-K时能启动；再开启后发一条短请求，随后32/128/1024输出逐步放大。
检查是否完成target图捕获、重放时预算是否变化，以及CPU任务是否正常完成。
日志前缀`Spec-K target_verify_stats`包含`tokens`、`top_k_sum`、`avg_top_k`和直方图。
关闭统计可用`log_stats_interval=0`。

统计是实际target生成验证位置的预算：排除prefill/padding，包含被拒绝位置、重复验证和末位完整预算。
默认prefill和draft仍为Top-6。7个位置全为3且末位为6时，完整8位置均值为3.375，不能要求日志恒为3。
保留完整层时，该值是动态层的预算，不是对所有层加权后的专家均值。

减少有效专家行有机会同时降低CPU/NPU专家工作与部分权重迁移；
固定形状的激活搬运、调度、等待和合并开销仍在，不保证所有通信字节或端到端耗时同比下降。
请使用相同设备、请求和固定输出长度对比；性能变化也可能伴随接受率及模型输出变化。

本次只新增一份小型[回归文件](../../tests/ut/spec_decode/test_spec_k_cpu_moe.py)，
覆盖预算、请求生命周期、两侧路由及迁移输入，并复用现有hybrid executor检查。
[验证记录](SPEC_K_CHECKS.md)区分CPU检查与尚未完成的A5验证。
