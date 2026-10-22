# DeepSeek V4.1 integer weight conversion for A3

This experimental CPU tool converts a native V4.1 FP4/FP8 checkpoint into
ModelSlim-format weights. It adapts the V4 conversion functions from
`rfc/vllm_cann` commit `aefc78918df625afbfa6f6f9ab00e4dd6f3b54a7`.
It requires the existing PyTorch and safetensors packages; it does not import
vLLM or TorchNPU and does not allocate an NPU device.

Conversion compatibility and model accuracy are separate checks. FP4 group-32
scales become INT4 per-channel scales, introducing additional quantization
error. This tool does not perform activation calibration, SmoothQuant or
QuaRot. Full-model loading, graph capture, accuracy and performance require
subsequent hardware tests.

## Conversion policy

| Source tensors | Output |
| --- | --- |
| Target routed experts, `layers.*.ffn.experts.*.w[123]` | INT4 W4A8, per-channel |
| Draft routed experts, `mtp.*.ffn.experts.*.w[123]` | INT8 W8A8, per-channel |
| Shared experts | INT8 W8A8 |
| Attention `wq_b`, `wo_b`, indexer `wq_b` | INT8 W8A8 |
| `wq_a`, `wkv`, `wo_a`, compressor and side projections | Decode FP8 to BF16 |
| Draft `main_proj` | Decode FP8 to BF16 |
| Vision FP8 matrices | Decode FP8 to BF16 |
| Engram lookup table and its E8M0 scales | Preserve original bytes |
| Other floating-point parameters, embeddings and heads | Preserve |

The FP8 decoder reads `weight_block_size` from the checkpoint, including
V4.1's 32-by-32 blocks. FP4 uses its separate group-32 format, and Engram tables
use per-row group-32 scales rather than two-dimensional blocks.

The V4.1 DSpark loader accepts BF16 `wo_a` directly. Its A3 `main_proj` is
constructed without a quantization config, so it must remain BF16. The V4
tool's rule that preserves native FP8 draft `wo_a`, and its rule that quantizes
`main_proj`, must not be copied into this recipe.

INT4 packing follows ModelSlim's output-channel ordering. Down-projection
`w2.scale_bias` has 16 columns, one compensation sum per input partition,
matching the existing MoE loader's `16 / moe_tp_size` allocation. Gate and up
projections have one compensation column. This also supports full expert
ownership with EP, where the effective MoE TP size is one.

## Audit first

Run inside the development container. No CANN environment setup or editable
package reinstall is needed to run this Python tool.

```bash
python3 examples/quantization/deepseek_v41/quantize.py \
  --input /home/models/DeepSeek-V4.1-Flash \
  --dry-run \
  --report /home/p00948912/dsv41_dev_20261019/runtime/v41_audit.json
```

This reads only JSON and safetensors headers. It checks expert projection and
layer coverage, DSpark layers, Engram weight/scale pairs, scale shapes and
that every indexed source tensor is consumed or preserved. Missing scales
raise an error instead of silently retaining unsupported packed weights.
The report includes projected output tensor bytes, excluding small file headers
and tokenizer assets.

## Convert a representative subset

```bash
python3 examples/quantization/deepseek_v41/quantize.py \
  --input /home/models/DeepSeek-V4.1-Flash \
  --output /home/p00948912/dsv41_dev_20261019/runtime/v41_sample \
  --include 'layers.6.ffn.experts.0.*' \
  --include 'mtp.0.attn.wo_a.weight' \
  --include 'mtp.0.main_proj.weight'
```

A subset includes each selected weight's source scale automatically. Its
output intentionally has no `config.json`, retains `CONVERSION_INCOMPLETE`,
and cannot be mistaken for a complete serving checkpoint.

## Full conversion

```bash
python3 examples/quantization/deepseek_v41/quantize.py \
  --input /home/models/DeepSeek-V4.1-Flash \
  --output /home/p00948912/DeepSeek-V4.1-Flash-int4-int8 \
  --threads 8 \
  --max-shard-mib 512
```

The source is read-only. The output must be a new or empty, non-overlapping
directory. Disk space is checked before conversion. This initial version does
not support resuming an interrupted conversion; unfinished output retains the
marker and is never silently overwritten.

Tensors are read individually, without loading entire source shards. Engram
tables are copied as raw safetensors byte ranges with an 8 MiB copy buffer,
preserving FP8 and E8M0 bits and avoiding whole-table BF16 expansion. This still
copies the table data to the new checkpoint and requires disk space and I/O
time. A table and its scale can exceed the ordinary output shard size limit.
The limit bounds the ordinary output buffer, not the largest decoded matrix
or quantization workspace.

On successful full conversion the output contains:

- `quant_model_weights-*.safetensors`;
- identical `model.safetensors.index.json` and
  `quant_model_weights.safetensors.index.json`, covering all output tensors;
- `quant_model_description.json` with the per-layer quantization schemes;
- `config.json` with explicit `quant_method: ascend`, preserving text, vision,
  DSpark and Engram configuration;
- tokenizer and other model assets;
- `conversion_report.json` with coverage and output byte counts.

After full conversion, a serving command must point to the **new checkpoint**
and use `--quantization ascend`; the original `deepseek_v4_fp8` flag would select
the native FP4 path that failed on this A3 stack. Engram host offload remains
controlled independently by `--engram-config '{"cpu_offload":true}'`.
Successful CPU conversion alone does not establish that eight devices can
finish model loading and graph capture.

`--target-expert-bits 8` is available as an alternative accuracy/control recipe.
It increases target expert storage and has not been established to fit eight
devices.

## CPU regression tests

From the Ascend repository, with this repository first on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m pytest \
  --confcutdir=tests/ut/quantization \
  tests/ut/quantization/test_deepseek_v41_conversion.py -q
```

These standalone tests exercise CPU math, cross-shard scales, output indices,
raw Engram byte copying, TP compensation, configuration and incomplete outputs.
They do not mock NPU computation or claim hardware validation.
