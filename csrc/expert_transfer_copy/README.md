# Device-owned expert byte copy (Ascend950PR)

The default AccOffloadSparseCopy ABI uses 2 AIV blocks and 64 KiB tiles. Source/destination pointers and lengths remain NPU tensors. No CPU expert decision callback or runtime weight conversion is introduced. Diagnostic copies use the same block count and tile size, with additional instrumentation fences.

Build from repository root after sourcing CANN:

    bash csrc/expert_transfer_copy/build.sh "$PWD/build/expert_transfer_copy"

This copies both shared libraries into vllm_ascend/expert_offload/lib. CPU MoE transfer_scheduler=npu resolves this library when transfer_copy_library is omitted or empty. Explicit paths override it. The default build target is ascend950pr_9599; other hardware is not validated by this benchmark.

There is no core-count, tile-size or pipeline-mode selector in the production copy path. Native MXFP4 conversion remains separate. Small tests cover full expert bytes, dynamic graph source/slot changes, inactive transfers, tail guards and diagnostic ring replay. Microbenchmark improvement is not a full-model TPOT claim.
