# V2 Spec-K 异步停滞修复及验证

更新时间：2026-10-23T09:55:39.444577+08:00

## 已验证的结果

已修改 V2 target 预算准备代码，去掉真实服务中的逐轮全设备同步，完成384条端到端与32并发profiling工作负载。全部8rank已解析，服务已自动关闭。当前证明范围是本次配对负载与回归用例；不能据单次跨设备数值称为稳定性能收益。

## 根因证据

1. 旧代码将动态Spec-K开启但阈值设为[1,1,1]，实际verify Top-K恒6，仍在32并发负载下停滞。该诊断服务的单请求1024预热完成，正式32并发超时。日志在../dsv41_v2_async_diagnosis_20261023_011026，说明不需要实际裁剪专家就能触发本次问题。
2. 单卡不加载模型、不调用MoE通信的最小链“draft预算图 → eager预算准备 → target裁剪图”也复现停滞。逐轮同步可掩盖；改为全部eager以及只回放其中一张图的短对照通过。
3. 编译IR定位到_prepare_budgets对History的离散写：当前后端为其生成hir.sync_block_lock/unlock，旧小规模产物lock_num=8。多次复现停在此预算准备之后。简单保留pinned副本、更换捕获stream、增加设备流等待、把prune的N静态化均未通过长验证；不是已验证的修复。
4. V2上游_prepare_pos_seq_lens_kernel明确生成每请求 num_computed_tokens + arange 的连续位置。利用同一真实NPU首位置展开连续地址后，History读写地址与原计算相同，但不再降低为离散写的软件锁路径。新产物无sync_block_lock，lock_num为空；160轮异步回放和完整模型负载通过。

精确到运行时锁缓冲区如何被复用的底层机制未另行证明；已验证的是这个kernel生成的软件锁执行路径与本次停滞的因果消融，以及规避它后的模型级结果。编译产物保存在runtime/spec_k_async_fix_20261023/compiler_evidence。

## 代码修改

- vllm_ascend/ops/triton/spec_decode/spec_k.py：增加CONTIGUOUS_POSITIONS编译期分支，V2按每tile的真实首位置展开连续History地址。
- vllm_ascend/spec_k.py：传递contiguous_positions能力，默认False，保留通用位置路径。
- vllm_ascend/worker/v2/model_runner.py：仅V2启用该能力；配置已禁止PP/PCP/DCP等不支持的Spec-K组合。
- 新增异步双图交错回归，轮间无synchronize，在最后一次等待后核对实际预算、路由权重和直方图；扩展interpreter对多tile、重排、历史重算和prompt掩码的验证。

动态预算算法、阈值、路由权重语义不变。未修改原生算子或安装依赖。现有editable安装可直接使用，需重启服务；Triton会编译新kernel。Baseline关闭Spec-K时不进入这条分支。

## 回归检查

- CPU合约：53 passed。
- 真实Triton CPU interpreter：39 passed。
- NPU：6 passed，包含统计开/关两组各160轮无逐轮同步的数值核对。
- 修改6个Python文件的Ruff check/format检查通过；git diff --check通过。

测试日志和最终patch：/home/p00948912/dsv41_dev_20261019/runtime/spec_k_async_fix_20261023

## Profiling（8rank各29个完整周期；ms/轮）

| 指标 | 原Baseline | 动态组诊断同步版 | 修复后无同步版 |
|---|---:|---:|---:|
| target图跨度 | 58.260933 | 55.689290 | 55.956492 |
| 其中40次融合MoE累计 | 24.012691 | 19.882732 | 20.170695 |
| target图外（含draft） | 12.636062 | 31.652015 | 13.109294 |
| 其中draft图跨度 | 6.516399 | 7.414095 | 6.971805 |
| 完整target-to-target周期 | 70.896996 | 87.341305 | 69.065787 |

子项不能重复相加。旧动态profiling在target前/后、sample/draft后都有全设备同步；本次没有这些运行时hook。rank0 trace中仅1次主线程全设备同步API，与采集控制相符，不再是每轮3次。

| 图外分段 | 原Baseline | 修复后 |
|---|---:|---:|
| target结束→draft开始 | 2.728023 | 2.703959 |
| draft图 | 6.516399 | 6.971805 |
| draft结束→下一target | 3.391641 | 3.433530 |

rank0已记录CPU算子区间中，与target重叠部分恢复为6.247586ms/轮（旧同步版约0.080319ms，Baseline约6.578509ms）。CPU区间为并集，非纯CPU指令耗时，不包含所有Python/IPC时间。

采集实际target平均Top-K=3.392968750；rank0 target主stream=89，draft主stream=277。

## 端到端（384条总请求，并发32，1024输出，ignore_eos=true）

| 指标 | 原Baseline | 旧动态入口同步版 | 修复后无同步版 |
|---|---:|---:|---:|
| 全程输出吞吐 token/s | 1280.491814 | 1348.530043 | 1617.790684 |
| 平均TPOT ms | 22.540550 | 21.918826 | 18.568524 |
| 平均TTFT ms | 1629.636685 | 720.186199 | 552.589623 |
| draft token接受率 % | 40.244622 | 48.840642 | 49.101309 |
| 每轮输出长度，含bonus | 3.817124 | 4.418845 | 4.437092 |

384/384成功，零失败，输出总数393216，输入总数86261。prompt内容与顺序与Baseline相同；按SSE token_ids校验每请求1024输出通过。

实际target平均Top-K=3.419747786。TPOT×报告的每轮输出长度=82.390241ms；直接按SSE返回区间加权=82.176449ms。Baseline相应近似值86.040064ms、直接值85.827479ms。这是全负载请求加权返回间隔，包含补位prefill、尾部并发变化和客户端传输，不能与稳态硬件69.065787ms要求相等。

维持并发窗口吞吐=1682.750227 token/s（旧动态1422.281337，Baseline1415.110664）；该窗口含补位prefill，不是纯decode平台。

## 比较限制与设备审计

本次设备为0、1、8–13；旧Baseline为0、1、6–11，旧入口同步动态为6–13，旧三同步profiling为8–15。该单次结果不能替代同卡重复配对性能结论。推理期间监测未发现外部进程占用；离线解析时卡0出现别的Python进程，发生在本服务已停止之后，详见resource_audit.json。未重置或停止他人设备/服务。

## 文件

- rank0/ASCEND_PROFILER_OUTPUT/trace_view.json：MindStudio Insight时间线；其他rank同结构。
- profile_inspection.json、rank0_step_coordinates.csv：完整周期与算子累计。
- outside_analysis.json、outside_dynamic_rank0.csv：图外分段与CPU重叠。
- e2e384/result.json、e2e_validation.json、actual_topk.json：端到端及真实预算统计。
- device_sync_audit.json、resource_audit.json、service_stopped.json：同步、资源及停服审计。
- start_server.sh、environment.sh、config.json：可复用启动配置。
