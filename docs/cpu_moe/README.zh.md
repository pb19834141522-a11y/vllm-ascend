# DeepSeek V4 Flash：9334 CPU/NPU 混合 MoE 使用说明

本文交付 CPU 计算后端、vLLM Ascend 适配、C++/AscendC 构建入口和启动流程。代码分别维护在 [yxk-fdu/vllm-ascend](https://github.com/yxk-fdu/vllm-ascend) 与 [yxk-fdu/ktransformers](https://github.com/yxk-fdu/ktransformers)。GitHub 用户原名 BeyondPro，现已改为 yxk-fdu。

## 1. 版本与路径先确认

9334 即 `90.90.93.34`，CPU 为 x86_64，NPU 为 Ascend950PR。当前容器名为 `cpu_dspark`。其他架构需重新编译，不可复制这里的 x86_64 `.so`；其他 SoC 的算子编译与运行需要单独验证。

| 组件 | 9334 实际运行环境（2026-10-20 读取） |
| --- | --- |
| Python | 3.11.10 |
| CANN | 9.1.0（构建解析到 `/usr/local/Ascend/cann-9.1.0`） |
| vLLM | 包版本 `0.30.0+empty`；源码基线 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，tag `v0.30.0` |
| vLLM Ascend | 包版本 `0.1.dev5559+g643ee9f69`；本次混合 MoE 改动的基线为 `643ee9f69585c87b1bd52ad9c3fd02221eaa70d3` |
| PyTorch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post2` |
| kt-kernel | ktransformers 基线 `d7b5b49a3ef214a822aba613423551dd56416557` 加本次 Ascend/MXFP4/异步运行补丁 |

`0.30.0+empty` 是本机源码构建的版本字符串，不是要求从 PyPI 安装同名 wheel。复现时固定上述 vLLM 提交，并使用两个仓库中本次配套的分支/提交。不要直接安装最新版或混用任意 vLLM Ascend main。9334 的 `/mnt/share/y50063808/vllm-ascend-workspace/vllm` 是 `v0.26.0`（`568afb3a13806beb53bb2e6bd518269357b237c0`），不是当前 `run_dpark.sh` 所使用的 vLLM；只切换 PYTHONPATH 到该目录会改变版本。

| 用途 | 9334 已有路径 |
| --- | --- |
| 启动脚本、辅助工具、专家热度文件 | `/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe` |
| 实际运行的 vLLM | `/mnt/data/y50063808/dspark/vllm` |
| 实际运行的 vLLM Ascend | `/mnt/data/y50063808/dspark/vllm-ascend` |
| 原 CPU 后端源码 | `/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/dependencies/ktransformers-AK/kt-kernel` |
| 当前 CPU 运行包 | `/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/kt_async_runtime` |
| 原生模型权重 | `/mnt/data/DeepSeek-V4-Flash-0731` |
| 分层 CPU GGUF 权重 | `/mnt/sdc/DeepSeek-V4-Flash-0731-GGUF` |
| GGUF 快捷入口 | `/mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/DeepSeek-V4-Flash-0731-GGUF`（指向上一行） |

**在 9334 上可以直接使用作者已经准备好的上述权重路径。其他服务器可能也有作者已经准备好的权重，请先联系作者确认路径，无需默认重新下载或转换。** `/mnt/share` 在 9334 是本地目录，不代表其他机器能看到相同文件。容器内也必须能看到这些绝对路径。

## 2. 执行流程与权重后端

路由产生专家 ID 和权重 → 根据当前 `log2phy` 分出 CPU/NPU 路由 → 提交 CPU 任务 → NPU 执行驻留专家 → 等待 CPU 完成 → 合并两侧输出。CPU/NPU 路由划分应以本步映射为准，避免额外静态掩码导致专家漏算。

| 启动入口 | CPU 权重 | NPU 权重 / 调度 |
| --- | --- | --- |
| `run_dpark.sh`（当前分享主入口） | 每层 `dsv4_layer{layer_idx}_mxfp4.gguf` | 原生模型加载后准备 NPU 使用的 NZ 权重；CPU 与 NPU 有独立权重表示；开启 dspark，7 个推测 token，以及 NPU 侧专家迁移调度 |
| `run_cpu_moe.sh`（原生共享内存路线） | `native_mxfp4`，借用 manager 的六组原生权重/scale buffer | 使用相同原生模型目录；当前脚本关闭专家迁移，不配置 dspark |

两条路线的脚本配置不同，不能只替换权重路径就认为相同。原生路线不需要先制作 GGUF。GGUF 路线仍需要完整原生模型目录，作为 tokenizer、配置、非专家权重与 NPU 权重的来源。

`num_device_layers`、`num_device_experts`、热专家文件共同影响驻留分配；前者不能简单解释为只有对应数量的层在 NPU 上执行。`CARDS=3` 表示物理 NPU 3，脚本的 TP/PP/DP 都为 1，不表示使用三张卡。

## 3. 权重准备

先复用作者路径，检查原生目录中的 `config.json`、tokenizer/encoding 文件、safetensors 分片及索引，以及 GGUF 路线各个实际需要的层文件。不要把单个 GGUF 当作完整模型。

如确实没有现成 GGUF，可使用计算后端仓库的转换器。它进行 MXFP4 位布局重排，保持已有量化编码与 scale，不是重新量化。以下命令一次转换一层；应依据模型配置和实际 offload 层范围重复执行，输出放在有足够容量的磁盘。

```bash
export MODEL_DIR=/mnt/data/DeepSeek-V4-Flash-0731
export GGUF_DIR=/your/disk/DeepSeek-V4-Flash-0731-GGUF
mkdir -p "$GGUF_DIR"
python "$KT_REPO/tools/convert_mxfp4_layer_to_gguf.py" \
  --input "$MODEL_DIR" --layer-idx 0 \
  --output "$GGUF_DIR/dsv4_layer0_mxfp4.gguf" --verify-reader
```

`KT_REPO` 指 ktransformers 仓库根目录。转换器依赖仓库锁定的 llama.cpp/gguf-py 及本次 MXFP4 补丁。NPU NZ 表示由模型加载与适配流程准备；这里不要求用户另外杜撰一个 NZ 权重目录。

## 4. 环境与源码安装

先准备与驱动匹配的 Ascend 容器、CANN、上述成对的 torch/torch_npu，以及 `numactl`、hwloc 开发包、pkg-config、GCC/G++、CMake、pybind11、setuptools、wheel、numpy、safetensors 和 gguf。9334 可进入现有容器：

```bash
docker exec -it cpu_dspark bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

新机器先分别 clone 两个仓库，并切到本次配套交付分支。用文档顶部固定的提交准备 vLLM。不要在有人使用的环境中直接重新安装，建议使用独立环境或容器。

```bash
git clone --branch share/cpu-moe-9334-20261020 https://github.com/yxk-fdu/vllm-ascend.git /your/work/vllm-ascend
git clone --branch share/cpu-moe-9334-20261020 https://github.com/yxk-fdu/ktransformers.git /your/work/ktransformers
git clone https://github.com/vllm-project/vllm.git /your/work/vllm
git -C /your/work/vllm checkout ced6857afa0ea7b2e3f0846a62e1394e90f15607
export ASCEND_REPO=/your/work/vllm-ascend
export VLLM_REPO=/your/work/vllm
export KT_REPO=/your/work/ktransformers
git -C "$KT_REPO" submodule update --init third_party/llama.cpp third_party/pybind11
git -C "$KT_REPO/third_party/llama.cpp" apply "$KT_REPO/patches/llama-cpp-mxfp4.patch"
cd "$VLLM_REPO"
VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-deps --no-build-isolation
cd "$ASCEND_REPO"
python -m pip install -e . --no-deps --no-build-isolation
```

`--no-deps` 的前提是依赖已在匹配容器中准备好，目的在于保留验证过的 torch/torch_npu 配对。它不能代替安装缺失的运行依赖。vLLM Ascend 的常规 CANN 自定义算子也必须完成本仓库原有构建，不能只编译下节两个独立算子就认为全部完成。

## 5. C++ 与 AscendC 编译

计算后端仓库提供 `kt-kernel/build_ascend_cpu_moe.sh`。它在隔离的构建目录生成 Python 包与辅助轮询库，不安装或覆盖当前在线服务的运行包。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash "$KT_REPO/kt-kernel/build_ascend_cpu_moe.sh" /your/work/kt-build
export KT_BUILD=/your/work/kt-build/package
export PYTHONPATH="$ASCEND_REPO:$VLLM_REPO:$KT_BUILD:${PYTHONPATH:-}"
export KT_MOE_POLL_LIBRARY="$KT_BUILD/kt_kernel/libkt_ready_poll.so"
python -c 'import torch,torch_npu,kt_kernel; print(kt_kernel.__file__); print(kt_kernel.kt_kernel_ext.__file__)'
```

9334 是 x86_64。构建参数使用本机 CPU 能力，不能把产物直接复制到 ARM 或不支持相同指令集的 x86 机器。Python ABI、torch/torch_npu、CANN 改变后也应重建。

在 vLLM Ascend 仓库构建 NPU 搬运算子和 CPU 结果合并算子：

```bash
cd "$ASCEND_REPO"
bash csrc/expert_transfer_copy/build.sh "$PWD/build/expert_transfer_copy" ascend950pr_9599
bash csrc/cpu_moe_mapped_combine/build.sh "$PWD/build/cpu_moe_mapped_combine" ascend950pr_9599
```

产物分别进入 `vllm_ascend/expert_offload/lib` 与 `vllm_ascend/expert_offload/cpu_backend/lib`。其他硬件需要确认自身 SoC 编译值并验证，不保证把上述值照搬即可运行。

## 6. 启动服务

本仓库 `examples/cpu_moe/9334/` 保存 9334 原有启动脚本、热专家排名和辅助脚本；这些是机器配置样例。脚本保留作者绝对路径及当前参数，便于在 9334 直接复用；在其他服务器上先复制脚本，再逐项修改路径、物理卡号、CPU/NUMA 绑定、端口、权重目录、库路径及模型内存参数。

### 9334 使用现有准备结果

检查 8150 是否已有服务以及目标卡是否有人使用，然后进入 `cpu_dspark`：

```bash
ss -ltn 'sport = :8150'
npu-smi info
docker exec -it cpu_dspark bash
bash /mnt/share/y50063808/vllm-ascend-workspace/cpu_moe/run_dpark.sh
```

当前 GGUF/dspark 样例使用物理卡 3、端口 8150、服务名 `deepseek_v4`、CPU worker 32 个，MoE 核 32–63、服务核 8–28 和 30–31、轮询/完成线程核 6/7、NUMA 0。这些绑定适配本机拓扑，不能直接当作通用模板。当前脚本含 45 项逐层驻留专家配置、`max_num_seqs=8`、`max_model_len=15000`、`FULL_DECODE_ONLY` 和 `num_speculative_tokens=7`；保留配套配置先验证，再按需求调参。

### 使用新编译的后端或迁移机器

复制 `examples/cpu_moe/9334/run_dpark.sh` 到自己的运行目录，更新脚本中的 `KT_BUILD`、`KT_MOE_POLL_LIBRARY`、`PYTHONPATH`、`ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH`、`transfer_copy_library`、GGUF 模板、原生模型目录及热专家文件。原脚本会重新赋值部分变量，仅在外部 export 不能保证覆盖。确认 `vllm` 命令来自目标 Python 环境，并打印实际模块路径：

```bash
which python vllm
python -m pip show vllm vllm-ascend torch torch-npu
python -c 'import vllm,vllm_ascend,kt_kernel; print(vllm.__file__); print(vllm_ascend.__file__); print(kt_kernel.__file__)'
```

原生路线的 `run_cpu_moe_host.sh` 会修改宿主 CPU/IRQ 策略并调用 `run_cpu_moe.sh`。它只适用于作者已经配置的机器，不能作为通用必跑步骤；共享机器需先协调 CPU/IRQ 使用。普通 GGUF 主入口不需要运行它。

## 7. 请求验证与评测

等服务完成权重加载和图捕获，再检查模型接口。以下请求会真正执行一次推理：

```bash
curl -fsS http://127.0.0.1:8150/v1/models
curl -fsS http://127.0.0.1:8150/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek_v4","messages":[{"role":"user","content":"请计算 1+1。"}],"max_tokens":64,"temperature":0}'
```

`examples/cpu_moe/9334/eval.sh` 保存作者的 ais-bench 性能与 GSM8K 评测命令；其中 `--models 8150` 是作者环境的 ais-bench 模型配置名，新环境还需准备对应 endpoint、服务名、tokenizer 和数据集配置，不能仅凭端口号自动得到该配置。

脚本默认在 `cpu_moe/cpu_profile/<时间戳>` 创建 profiler 目录与 CPU 计算日志；服务 stdout 默认不自动落盘。需要服务日志时可用 `bash run_dpark.sh 2>&1 | tee service.log`，脚本的 pipefail 会保留服务失败状态。

## 8. 常见定位

- 找不到 kt_kernel / 未定义符号：核对 PYTHONPATH、实际 `.so` 路径、Python ABI 和 torch/torch_npu 配对；在目标容器内检查依赖并重新编译。
- GGUF 层文件不存在：核对模板中的 `{layer_idx}`、符号链接目标和容器挂载；换机器先向作者问现成权重路径。
- 缺少 NPU 搬运或合并库：按第 5 节编译，检查启动脚本是否仍写着旧源码目录。
- CPU affinity / NUMA 失败：检查 `lscpu -e`、`numactl -H` 和容器 cpuset；按实际拓扑调整。
- vLLM 参数或导入失败：先排除误用了 workspace v0.26.0，使用实际运行的 v0.30.0 基线及配套 Ascend 分支。

本次交付的静态检查、重新编译与测试结果单独记录在 `VALIDATION.md`；文档中的推理请求是使用者验证步骤，不代表本次更新重新启动过完整模型或重新跑过精度评测。
