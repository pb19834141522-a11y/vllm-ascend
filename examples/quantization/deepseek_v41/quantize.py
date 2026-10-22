# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline V4.1 FP4/FP8 to ModelSlim INT4/INT8 conversion, using CPU only.

The quantizers derive from examples/quantization/deepseek_v4/quantize.py at
vllm-project/vllm-ascend rfc/vllm_cann commit aefc78918df625afbfa6f6f9ab00e4dd6f3b54a7.
This adapter handles V4.1 block sizes, draft loading and indexed Engram tables.
It does not calibrate activations or establish model-level accuracy.
"""

import argparse
import copy
import fnmatch
import json
import math
import os
import re
import shutil
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_GROUP_SIZE = 32
INT4_BIAS_PARTITIONS = 16
QUANT_EPSILON = 1e-5
COPY_BUFFER_BYTES = 8 * 1024 * 1024
HEADER_LIMIT_BYTES = 100 * 1024 * 1024
MIB = 1024 * 1024
FLOAT = "FLOAT"
W4A8 = "W4A8_DYNAMIC"
W8A8 = "W8A8_DYNAMIC"
FP8_DTYPES = frozenset({"F8_E4M3", "F8_E4M3FN"})
EXPERT_PATTERN = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")
QUANT_LINEAR_PATTERN = re.compile(
    r"^(layers|mtp)\.\d+\.(ffn\.shared_experts\.w[123]|attn\.(wq_b|wo_b|indexer\.wq_b))\.weight$"
)


@dataclass(frozen=True)
class TensorInfo:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    offsets: tuple[int, int]

    @property
    def nbytes(self):
        return self.offsets[1] - self.offsets[0]


@dataclass(frozen=True)
class Conversion:
    name: str
    scheme: str
    scale: str | None = None
    preserve_table: bool = False

    @property
    def inputs(self):
        return (self.name, self.scale) if self.scale else (self.name,)


def read_inventory(model: Path) -> tuple[dict, dict[str, TensorInfo]]:
    """Read headers only, including for the approximately 95 GiB Engram shards."""
    index = json.loads((model / "model.safetensors.index.json").read_text())
    headers = {}
    for filename in sorted(set(index["weight_map"].values())):
        path = (model / filename).resolve()
        if not path.is_relative_to(model.resolve()):
            raise ValueError(f"Checkpoint shard outside model directory: {filename}")
        with path.open("rb") as handle:
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            size = struct.unpack("<Q", size_bytes)[0]
            if size > HEADER_LIMIT_BYTES:
                raise ValueError(f"Safetensors header too large: {path}")
            header = json.loads(handle.read(size))
        headers[filename] = (path, size + 8, header)
    inventory = {}
    for name, filename in index["weight_map"].items():
        path, start, header = headers[filename]
        if name not in header:
            raise ValueError(f"Index tensor {name} absent from {filename}")
        item = header[name]
        lo, hi = item["data_offsets"]
        if not 0 <= lo <= hi <= path.stat().st_size - start:
            raise ValueError(f"Invalid tensor byte range: {name}")
        inventory[name] = TensorInfo(path, item["dtype"], tuple(item["shape"]), start, (lo, hi))
    return json.loads((model / "config.json").read_text()), inventory


def fp8_block_size(config: dict) -> tuple[int, int]:
    text = config.get("text_config", config)
    quant = config.get("quantization_config", text.get("quantization_config", {}))
    if quant.get("quant_method") != "fp8":
        raise ValueError("Expected the original FP4/FP8 checkpoint, not an already converted checkpoint")
    block = quant.get("weight_block_size")
    if not isinstance(block, list) or len(block) != 2 or any(type(value) is not int or value <= 0 for value in block):
        raise ValueError("quantization_config.weight_block_size must contain two positive integers")
    return tuple(block)


def decode_fp8(weight: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("Block FP8 decoding requires a matrix")
    rows, cols = weight.shape
    expected = (math.ceil(rows / block[0]), math.ceil(cols / block[1]))
    if tuple(scale.shape) != expected:
        raise ValueError(f"FP8 scale shape {tuple(scale.shape)} != {expected} for block {block}")
    expanded = scale.float().repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    return (weight.float() * expanded[:rows, :cols]).bfloat16()


def decode_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[1] * 2 % FP4_GROUP_SIZE:
        raise ValueError("Packed FP4 matrix must have a logical input size divisible by 32")
    expected = (weight.shape[0], weight.shape[1] * 2 // FP4_GROUP_SIZE)
    if tuple(scale.shape) != expected:
        raise ValueError(f"FP4 scale shape {tuple(scale.shape)} != {expected}")
    raw = weight.view(torch.uint8)
    codes = torch.stack([raw & 15, raw >> 4], dim=-1).flatten(-2)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[codes.long() & 7]
    values *= 1.0 - 2.0 * ((codes >> 3) & 1).float()
    return (values * scale.float().repeat_interleave(FP4_GROUP_SIZE, -1)).bfloat16()


def pack_int4(weight: torch.Tensor) -> torch.Tensor:
    """ModelSlim packs adjacent output channels, unlike native FP4's K packing."""
    if weight.ndim != 2 or weight.shape[0] % 2:
        raise ValueError("INT4 packing requires an even output dimension")
    rows, cols = weight.shape
    pairs = weight.T.contiguous().reshape(-1, 2)
    packed = (pairs[:, 0] & 15) | (pairs[:, 1] << 4)
    return packed.reshape(cols, rows // 2).T.contiguous().to(torch.int8)


def quantize_per_channel(weight: torch.Tensor, bits: int, down_projection: bool = False) -> dict[str, torch.Tensor]:
    if bits not in (4, 8) or weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("Quantization requires a finite matrix and bits=4 or bits=8")
    weight = weight.float()
    qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    scale = torch.maximum(weight.amin(-1).abs(), weight.amax(-1).abs()).clamp_min(QUANT_EPSILON)[:, None] / qmax
    codes = torch.round(weight / scale).clamp(qmin, qmax).to(torch.int8)
    result = {"weight": codes, "weight_scale": scale, "weight_offset": torch.zeros_like(scale)}
    if bits == 4:
        result["weight"] = pack_int4(codes)
        # The real ModelSlim MoE loader allocates [hidden, 16/tp] for w2.
        # A single summed column would broadcast and multiply compensation by 16.
        partitions = INT4_BIAS_PARTITIONS if down_projection else 1
        if weight.shape[1] % partitions:
            raise ValueError(f"w2 input size must be divisible by {partitions}")
        dequantized = codes.float() * scale
        result["scale_bias"] = dequantized.reshape(weight.shape[0], partitions, -1).sum(-1) * 8
    return result


def validate_config_coverage(config: dict, inventory: dict[str, TensorInfo]):
    text = config.get("text_config", config)
    for layer in text.get("engram_layer_ids", []):
        weight = f"layers.{layer}.engram.embed.weight"
        if weight not in inventory or weight.removesuffix(".weight") + ".scale" not in inventory:
            raise ValueError(f"Missing Engram table or scale for layer {layer}")
    if text.get("dspark_block_size"):
        expected = set(range(text.get("num_nextn_predict_layers", 3)))
        actual = {int(name.split(".")[1]) for name in inventory if name.startswith("mtp.")}
        if actual != expected:
            raise ValueError(f"DSpark layer IDs {sorted(actual)} != {sorted(expected)}")
    expected_parts = {"1", "2", "3"}
    parts = {}
    for name in inventory:
        if match := EXPERT_PATTERN.fullmatch(name):
            tree, layer, expert, part = match.groups()
            parts.setdefault((tree, int(layer), int(expert)), set()).add(part)
    for expert, found in parts.items():
        if found != expected_parts:
            raise ValueError(f"Incomplete expert {expert}: found projections {found}")
    for tree, layer_count, expert_count in (
        ("layers", text.get("num_hidden_layers", 0), text.get("n_routed_experts", 0)),
        (
            "mtp",
            text.get("num_nextn_predict_layers", 3) if text.get("dspark_block_size") else 0,
            text.get("dspark_n_routed_experts", text.get("n_routed_experts", 0)),
        ),
    ):
        if expert_count:
            for layer in range(layer_count):
                found = {expert for kind, idx, expert in parts if kind == tree and idx == layer}
                if found != set(range(expert_count)):
                    raise ValueError(f"Incomplete {tree}.{layer} expert inventory: {len(found)} of {expert_count}")


def build_plan(config: dict, inventory: dict[str, TensorInfo], target_bits: int = 4) -> list[Conversion]:
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("This recipe is specifically for deepseek_v41")
    block = fp8_block_size(config)
    validate_config_coverage(config, inventory)
    plan, consumed = [], set()
    for name, info in inventory.items():
        if name.endswith(".scale"):
            continue
        scale_name = name.removesuffix(".weight") + ".scale"
        scale = scale_name if name.endswith(".weight") and scale_name in inventory else None
        table = name.endswith(".engram.embed.weight")
        scheme = FLOAT
        if match := EXPERT_PATTERN.fullmatch(name):
            scheme = W4A8 if match[1] == "layers" and target_bits == 4 else W8A8
        elif QUANT_LINEAR_PATTERN.fullmatch(name):
            scheme = W8A8
        # V4.1 A3's main_proj is explicitly constructed without quant_config.
        # wo_a, wq_a, wkv and compressor/indexer side matrices stay BF16.
        if table:
            if info.dtype not in FP8_DTYPES or scale is None:
                raise ValueError(f"Expected native MXFP8 Engram table with scales: {name}")
            if info.shape[-1] % FP4_GROUP_SIZE:
                raise ValueError(f"Engram row width must be divisible by 32: {name}")
            expected = (info.shape[0], info.shape[1] // FP4_GROUP_SIZE)
            if inventory[scale].shape != expected or inventory[scale].dtype != "F8_E8M0":
                raise ValueError(f"Invalid Engram group-32 E8M0 scales: {name}")
        elif info.dtype in FP8_DTYPES:
            expected = tuple(math.ceil(size / tile) for size, tile in zip(info.shape, block))
            if len(info.shape) != 2 or scale is None or inventory[scale].shape != expected:
                raise ValueError(f"Missing or incompatible block FP8 scales for {name}")
        elif info.dtype in ("I8", "U8") and scale:
            expected = (info.shape[0], info.shape[1] * 2 // FP4_GROUP_SIZE)
            if (
                not EXPERT_PATTERN.fullmatch(name)
                or info.shape[1] * 2 % FP4_GROUP_SIZE
                or inventory[scale].shape != expected
            ):
                raise ValueError(f"Unrecognized packed FP4 tensor {name}")
        elif scale:
            raise ValueError(f"Unexpected quantization scale for {name} ({info.dtype})")
        if scale and inventory[scale].dtype not in ("F8_E8M0", "F32", "BF16"):
            raise ValueError(f"Unsupported scale dtype for {name}: {inventory[scale].dtype}")
        if scheme != FLOAT:
            logical_rows = info.shape[0]
            if len(info.shape) != 2 or (scheme == W4A8 and logical_rows % 2):
                raise ValueError(f"Invalid quantized matrix shape: {name}")
        conversion = Conversion(name, scheme, scale, table)
        plan.append(conversion)
        consumed.update(conversion.inputs)
    if missing := set(inventory) - consumed:
        raise ValueError(f"Unconsumed source tensors (including orphan scales): {sorted(missing)[:10]}")
    return plan


def output_nbytes(entry: Conversion, inventory: dict[str, TensorInfo]) -> int:
    info = inventory[entry.name]
    if entry.preserve_table:
        return sum(inventory[name].nbytes for name in entry.inputs)
    logical_shape = info.shape
    if info.dtype in ("I8", "U8") and entry.scale:
        logical_shape = (info.shape[0], info.shape[1] * 2)
    count = math.prod(logical_shape)
    if entry.scheme == FLOAT:
        return count * 2 if entry.scale else info.nbytes
    rows = logical_shape[0]
    if entry.scheme == W8A8:
        return count + rows * 2 * 4
    bias_cols = INT4_BIAS_PARTITIONS if entry.name.endswith(".w2.weight") else 1
    return count // 2 + rows * (2 + bias_cols) * 4


def copy_raw_tensors(destination: Path, entries: dict[str, TensorInfo]):
    """Copy safetensor byte ranges without decoding or materializing huge tables."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, info in entries.items():
        header[name] = {"dtype": info.dtype, "shape": list(info.shape), "data_offsets": [offset, offset + info.nbytes]}
        offset += info.nbytes
    serialized = json.dumps(header, separators=(",", ":")).encode()
    serialized += b" " * (-len(serialized) % 8)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("xb") as output:
        output.write(struct.pack("<Q", len(serialized)))
        output.write(serialized)
        for info in entries.values():
            with info.path.open("rb") as source:
                source.seek(info.data_start + info.offsets[0])
                remaining = info.nbytes
                while remaining:
                    chunk = source.read(min(remaining, COPY_BUFFER_BYTES))
                    if not chunk:
                        raise ValueError(f"Truncated tensor data in {info.path}")
                    output.write(chunk)
                    remaining -= len(chunk)
    os.replace(temporary, destination)


class ShardWriter:
    def __init__(self, directory: Path, limit_bytes: int):
        self.directory = directory
        self.limit_bytes = limit_bytes
        self.pending = {}
        self.pending_bytes = 0
        self.weight_map = {}
        self.description = {"version": "1.0.0", "group_size": 0, "model_quant_type": W4A8}
        self.total_bytes = 0
        self.shard_count = 0

    def next_path(self):
        self.shard_count += 1
        return self.directory / f"quant_model_weights-{self.shard_count:05d}.safetensors"

    def flush(self):
        if self.pending:
            path = self.next_path()
            temporary = path.with_suffix(".tmp")
            save_file(self.pending, str(temporary), metadata={"format": "pt"})
            os.replace(temporary, path)
            self.weight_map.update(dict.fromkeys(self.pending, path.name))
            self.pending.clear()
            self.pending_bytes = 0

    def add(self, payload: dict[str, torch.Tensor], scheme: str):
        if set(payload) & self.description.keys():
            raise ValueError("Duplicate converted tensor")
        size = sum(tensor.numel() * tensor.element_size() for tensor in payload.values())
        if self.pending_bytes + size > self.limit_bytes:
            self.flush()
        self.pending.update({name: tensor.detach().contiguous() for name, tensor in payload.items()})
        self.description.update(dict.fromkeys(payload, scheme))
        self.total_bytes += size
        self.pending_bytes += size
        if self.pending_bytes >= self.limit_bytes:
            self.flush()

    def add_raw(self, entries: dict[str, TensorInfo]):
        self.flush()
        if set(entries) & self.description.keys():
            raise ValueError("Duplicate preserved tensor")
        path = self.next_path()
        copy_raw_tensors(path, entries)
        self.weight_map.update(dict.fromkeys(entries, path.name))
        self.description.update(dict.fromkeys(entries, FLOAT))
        self.total_bytes += sum(info.nbytes for info in entries.values())

    def close(self):
        self.flush()
        index = {"metadata": {"total_size": self.total_bytes}, "weight_map": self.weight_map}
        # Both loaders must see exactly the same tensor map. Engram checks the
        # quant index first; vLLM's generic safetensor filter reads the model index.
        for filename in ("model.safetensors.index.json", "quant_model_weights.safetensors.index.json"):
            (self.directory / filename).write_text(json.dumps(index, indent=2) + "\n")
        (self.directory / "quant_model_description.json").write_text(json.dumps(self.description, indent=2) + "\n")


def read_tensor(info: TensorInfo, name: str):
    with safe_open(str(info.path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def converted_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("quantization_config", None)
    if "text_config" in result:
        result["text_config"].pop("quantization_config", None)
    # Explicit format metadata prevents auto-selecting the original native FP4
    # backend. Per-layer schemes still come from quant_model_description.json.
    result["quantization_config"] = {"quant_method": "ascend"}
    return result


def copy_assets(model: Path, output: Path):
    for root, directories, filenames in os.walk(model):
        directories[:] = [name for name in directories if not name.startswith(".")]
        for filename in filenames:
            if filename in ("config.json", "quant_model_description.json") or filename.endswith(
                ".safetensors.index.json"
            ):
                continue
            if Path(filename).suffix not in (".json", ".py", ".jinja", ".txt", ".model", ".tiktoken"):
                continue
            source = Path(root) / filename
            destination = output / source.relative_to(model)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def convert_checkpoint(
    model: Path,
    output: Path | None = None,
    *,
    dry_run: bool = False,
    include: tuple[str, ...] = (),
    target_bits: int = 4,
    max_shard_bytes: int = 512 * MIB,
) -> dict:
    model = model.resolve()
    if target_bits not in (4, 8) or max_shard_bytes <= 0:
        raise ValueError("target_bits must be 4 or 8, shard size must be positive")
    config, inventory = read_inventory(model)
    plan = build_plan(config, inventory, target_bits)
    selected = [
        entry for entry in plan if not include or any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in include)
    ]
    if not selected:
        raise ValueError("No tensors matched --include")
    text = config.get("text_config", config)
    report = {
        "source": str(model),
        "dry_run": dry_run,
        "complete_selection": len(selected) == len(plan),
        "source_tensor_count": len(inventory),
        "selected_operations": len(selected),
        "consumed_source_tensors": sum(len(entry.inputs) for entry in selected),
        "fp8_block_size": list(fp8_block_size(config)),
        "dspark_block_size": text.get("dspark_block_size"),
        "engram_layer_ids": text.get("engram_layer_ids", []),
        "target_expert_bits": target_bits,
        "draft_expert_bits": 8,
        "operation_counts": dict(
            Counter(entry.scheme if not entry.preserve_table else "ENGRAM_PRESERVE" for entry in selected)
        ),
        "projected_tensor_bytes": sum(output_nbytes(entry, inventory) for entry in selected),
        "examples": {
            scheme: [entry.name for entry in selected if entry.scheme == scheme][:8] for scheme in (W4A8, W8A8, FLOAT)
        },
        "accuracy_validated": False,
        "npu_execution_validated": False,
    }
    if dry_run:
        return report
    if output is None:
        raise ValueError("--output is required for conversion")
    output = output.resolve()
    if output.is_relative_to(model) or model.is_relative_to(output):
        raise ValueError("Input and output directories must not overlap")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be an empty directory; existing results are never overwritten")
    existing_parent = output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    required_bytes = report["projected_tensor_bytes"] + max_shard_bytes
    if shutil.disk_usage(existing_parent).free < required_bytes:
        raise ValueError(f"Insufficient free disk space for {required_bytes} bytes of output and shard headroom")
    output.mkdir(parents=True, exist_ok=True)
    # Keep an interrupted conversion visibly incomplete. config.json is only
    # published after all weights, indices and assets have been written.
    marker = output / "CONVERSION_INCOMPLETE"
    marker.write_text("Do not serve this directory until conversion finishes.\n")
    writer = ShardWriter(output, max_shard_bytes)
    writer.description["model_quant_type"] = W4A8 if target_bits == 4 else W8A8
    block = fp8_block_size(config)
    for idx, entry in enumerate(selected):
        info = inventory[entry.name]
        if entry.preserve_table or (entry.scheme == FLOAT and not entry.scale and info.nbytes >= max_shard_bytes):
            writer.add_raw({name: inventory[name] for name in entry.inputs})
        else:
            weight = read_tensor(info, entry.name)
            if entry.scale:
                scale = read_tensor(inventory[entry.scale], entry.scale)
                weight = decode_fp8(weight, scale, block) if info.dtype in FP8_DTYPES else decode_fp4(weight, scale)
                if not torch.isfinite(weight).all():
                    raise ValueError(f"Nonfinite decoded values: {entry.name}")
            if entry.scheme == FLOAT:
                writer.add({entry.name: weight}, FLOAT)
            else:
                bits = 4 if entry.scheme == W4A8 else 8
                payload = quantize_per_channel(weight, bits, down_projection=entry.name.endswith(".w2.weight"))
                prefix = entry.name.removesuffix(".weight")
                writer.add({prefix + "." + suffix: tensor for suffix, tensor in payload.items()}, entry.scheme)
        if idx % 1000 == 0 or idx + 1 == len(selected):
            print(f"Converted {idx + 1}/{len(selected)}: {entry.name}", flush=True)
    writer.close()
    report.update(
        {
            "output_tensor_count": len(writer.weight_map),
            "output_tensor_bytes": writer.total_bytes,
            "output_shards": writer.shard_count,
            "conversion_finished": True,
        }
    )
    if writer.total_bytes != report["projected_tensor_bytes"]:
        raise ValueError("Written tensor size differs from preflight estimate")
    if report["complete_selection"]:
        copy_assets(model, output)
        (output / "config.json").write_text(json.dumps(converted_config(config), indent=2) + "\n")
        marker.unlink()
    else:
        marker.write_text("Partial conversion for inspection only; config.json intentionally omitted.\n")
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original V4.1 FP4/FP8 checkpoint")
    parser.add_argument("--output", type=Path, help="New empty output directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Read headers and validate coverage; no weight conversion"
    )
    parser.add_argument(
        "--include", action="append", default=[], help="Select checkpoint tensor globs; output is partial"
    )
    parser.add_argument("--target-expert-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--max-shard-mib", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", type=Path, help="Additional audit report path")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    report = convert_checkpoint(
        args.input,
        args.output,
        dry_run=args.dry_run,
        include=tuple(args.include),
        target_bits=args.target_expert_bits,
        max_shard_bytes=args.max_shard_mib * MIB,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
