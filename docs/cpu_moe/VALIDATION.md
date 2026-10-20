# 9334 交付验证（2026-10-20）

- 环境：cpu_dspark，Python 3.11.10，torch 2.10.0+cpu，torch_npu 2.10.0.post2，CANN 9.1.0。
- kt-kernel：独立 CMake 构建成功；新包可导入，launch_no_block_callback 符号存在；轮询辅助库编译成功。
- Ascend950PR：expert_transfer_copy 和 cpu_moe_mapped_combine 的 AscendC kernel 与 C++ dispatch 库均重新编译成功。
- 8 个相关适配单测文件：90 passed；覆盖 CPU 后端接线、路由拆分、专家迁移、原生权重和图完成协议。使用隔离编译的 CPU 包。
- 已更新旧 callback/H2D 协议测试，覆盖 mapped-host 协议的提交分支合流、ExternalEvent 完成等待、轮询/异步提交、非法旧路径和无效事件拒绝。
- llama.cpp MXFP4 补丁：针对锁定提交六个文件生成，反向应用检查通过。
- git diff --check 与构建脚本 bash -n 通过；交付 Python 文件均通过 AST 解析。
- 仓库要求的 bash format.sh ci 已尝试，因 cpu_dspark 未安装 pre-commit 无法执行完整 lint；未宣称格式检查全部通过。
- 未重启在线模型服务，未重新执行整模型推理、GSM8K 精度或性能评测；不对跨硬件支持或性能收益作额外保证。

编译日志在 9334 独立交付目录 `/mnt/data/y50063808/dspark/share-delivery-20261020`，不作为运行时依赖。没有复制现有二进制作为源码交付，也没有改变现有服务的启动路径。
