# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checkpoint conversion tests; no model runner or NPU allocation."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# vLLM also ships a regular "examples" package. Load the standalone converter
# by path so its package cannot shadow this repository's examples namespace.
SPEC = importlib.util.spec_from_file_location(
    "v41_converter_test", Path(__file__).resolve().parents[3] / "examples/quantization/deepseek_v41/quantize.py"
)
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)
FLOAT, W4A8, W8A8 = CONVERTER.FLOAT, CONVERTER.W4A8, CONVERTER.W8A8
build_plan = CONVERTER.build_plan
convert_checkpoint = CONVERTER.convert_checkpoint
copy_raw_tensors = CONVERTER.copy_raw_tensors
decode_fp4 = CONVERTER.decode_fp4
decode_fp8 = CONVERTER.decode_fp8
fp8_block_size = CONVERTER.fp8_block_size
pack_int4 = CONVERTER.pack_int4
quantize_per_channel = CONVERTER.quantize_per_channel
read_inventory = CONVERTER.read_inventory


def e8m0_ones(shape):
    return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def unpack_int4(packed):
    raw = packed.view(torch.uint8)
    result = torch.stack([raw & 15, raw >> 4], dim=1).reshape(packed.shape[0] * 2, packed.shape[1]).int()
    return torch.where(result >= 8, result - 16, result).to(torch.int8)


@pytest.fixture
def source(tmp_path):
    model = tmp_path / "native"
    model.mkdir()
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "expert_dtype": "fp4"},
        "text_config": {
            "num_hidden_layers": 1,
            "n_routed_experts": 1,
            "engram_layer_ids": [1],
            "dspark_block_size": 5,
            "num_nextn_predict_layers": 3,
            "dspark_n_routed_experts": 1,
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]},
        },
        "vision_config": {"hidden_size": 32},
    }
    weights, scales = {}, {}
    generator = torch.Generator().manual_seed(41)

    def fp8(name, shape, table=False):
        weights[name + ".weight"] = torch.randn(shape, generator=generator).to(torch.float8_e4m3fn)
        scale_shape = (shape[0], shape[1] // 32) if table else ((shape[0] + 31) // 32, (shape[1] + 31) // 32)
        scales[name + ".scale"] = e8m0_ones(scale_shape)

    for prefix in ("layers.0", "mtp.0", "mtp.1", "mtp.2"):
        for part, shape in (("w1", (32, 64)), ("w3", (32, 64)), ("w2", (64, 32))):
            name = f"{prefix}.ffn.experts.0.{part}"
            weights[name + ".weight"] = torch.randint(
                -128, 128, (shape[0], shape[1] // 2), dtype=torch.int8, generator=generator
            )
            scales[name + ".scale"] = e8m0_ones((shape[0], shape[1] // 32))
        fp8(prefix + ".attn.wo_a", (64, 32))
        fp8(prefix + ".attn.wq_b", (64, 32))
        weights[prefix + ".ffn.gate.weight"] = torch.randn((1, 64), generator=generator).bfloat16()
    for part, shape in (("w1", (32, 64)), ("w3", (32, 64)), ("w2", (64, 32))):
        fp8(f"layers.0.ffn.shared_experts.{part}", shape)
    fp8("layers.0.attn.wq_a", (32, 64))
    fp8("mtp.0.main_proj", (64, 192))
    fp8("layers.1.engram.embed", (6, 64), table=True)
    fp8("layers.1.engram.wkv", (64, 64))
    fp8("vision.blocks.0.attn.wqkv", (64, 32))
    weights["layers.1.engram.q_weight"] = torch.ones((4, 64), dtype=torch.bfloat16)
    weights["layers.1.engram.k_weight"] = torch.ones((4, 64), dtype=torch.bfloat16)
    weights["head.weight"] = torch.ones((128, 64), dtype=torch.bfloat16)
    weights["embed.weight"] = torch.ones((128, 64), dtype=torch.bfloat16)
    # Keep scales in another shard to catch readers that assume colocated pairs.
    save_file(weights, str(model / "weights.safetensors"))
    save_file(scales, str(model / "scales.safetensors"))
    index = {
        "weight_map": {**dict.fromkeys(weights, "weights.safetensors"), **dict.fromkeys(scales, "scales.safetensors")}
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    (model / "config.json").write_text(json.dumps(config))
    (model / "tokenizer_config.json").write_text('{"tokenizer_class":"Example"}')
    return model


@pytest.mark.parametrize("block", [(32, 32), (128, 128)])
def test_block_fp8_decoding_and_nondivisible_edges(block):
    rows, cols = block[0] + 3, block[1] * 2 + 5
    weight = torch.ones((rows, cols)).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1, 2, 4], [8, 16, 32]], dtype=torch.float32)
    output = decode_fp8(weight, scale, block)
    assert output[0, 0] == 1
    assert output[0, block[1]] == 2
    assert output[-1, -1] == 32
    assert output.shape == weight.shape


def test_fp8_rejects_wrong_block_scale():
    with pytest.raises(ValueError, match="scale shape"):
        decode_fp8(torch.ones((64, 64)).to(torch.float8_e4m3fn), torch.ones((1, 1)), (32, 32))


def test_fp4_all_codes_and_scale_groups():
    codes = torch.arange(16, dtype=torch.uint8).repeat(4).reshape(1, 64)
    packed = ((codes[:, 0::2] & 15) | (codes[:, 1::2] << 4)).view(torch.int8)
    scales = torch.tensor([[1.0, 2.0]])
    actual = decode_fp4(packed, scales)
    positive = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    expected = torch.cat([positive, -positive]).repeat(4).reshape(1, 64)
    expected[:, 32:] *= 2
    torch.testing.assert_close(actual.float(), expected)


def test_int4_packing_uses_output_dimension():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(32).reshape(16, 32)
    packed = pack_int4(values)
    assert packed.shape == (8, 32)
    assert torch.equal(unpack_int4(packed), values)
    with pytest.raises(ValueError, match="even output"):
        pack_int4(torch.ones((3, 32), dtype=torch.int8))


@pytest.mark.parametrize("bits", [4, 8])
def test_zero_weight_quantization_is_finite(bits):
    tensors = quantize_per_channel(torch.zeros((32, 64)), bits, down_projection=True)
    assert not tensors["weight"].any()
    assert torch.isfinite(tensors["weight_scale"]).all()
    assert (tensors["weight_scale"] > 0).all()
    with pytest.raises(ValueError, match="finite"):
        quantize_per_channel(torch.full((32, 64), float("nan")), bits)


def test_w2_compensation_matches_each_tp_shard():
    torch.manual_seed(41)
    tensors = quantize_per_channel(torch.randn(32, 128), 4, down_projection=True)
    bias = tensors["scale_bias"]
    decoded = unpack_int4(tensors["weight"]).float() * tensors["weight_scale"]
    assert bias.shape == (32, 16)
    for tp_size in (1, 2, 4, 8, 16):
        for rank in range(tp_size):
            reference = decoded.chunk(tp_size, dim=1)[rank].sum(-1) * 8
            actual = bias.chunk(tp_size, dim=1)[rank].sum(-1)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-5)


def test_nested_config_and_v41_draft_policy(source):
    config, inventory = read_inventory(source)
    assert fp8_block_size(config) == (32, 32)
    plan = {entry.name: entry for entry in build_plan(config, inventory)}
    assert plan["layers.0.ffn.experts.0.w1.weight"].scheme == W4A8
    assert plan["mtp.0.ffn.experts.0.w1.weight"].scheme == W8A8
    assert plan["layers.0.ffn.shared_experts.w1.weight"].scheme == W8A8
    assert plan["layers.0.attn.wq_b.weight"].scheme == W8A8
    assert plan["layers.0.attn.wq_a.weight"].scheme == FLOAT
    assert plan["mtp.0.main_proj.weight"].scheme == FLOAT
    assert all(plan[f"mtp.{idx}.attn.wo_a.weight"].scheme == FLOAT for idx in range(3))
    assert plan["layers.1.engram.embed.weight"].preserve_table
    assert plan["vision.blocks.0.attn.wqkv.weight"].scheme == FLOAT
    config["text_config"]["num_nextn_predict_layers"] = 4
    with pytest.raises(ValueError, match="DSpark layer IDs"):
        build_plan(config, inventory)


def test_missing_scales_and_missing_experts_fail_preflight(source):
    config, inventory = read_inventory(source)
    incomplete = dict(inventory)
    incomplete.pop("layers.0.attn.wq_b.scale")
    with pytest.raises(ValueError, match="FP8 scales"):
        build_plan(config, incomplete)
    incomplete = dict(inventory)
    incomplete.pop("layers.0.ffn.experts.0.w2.weight")
    with pytest.raises(ValueError, match="Incomplete expert"):
        build_plan(config, incomplete)
    incomplete = dict(inventory)
    incomplete["orphan.scale"] = next(iter(inventory.values()))
    with pytest.raises(ValueError, match="Unconsumed"):
        build_plan(config, incomplete)


def test_engram_raw_copy_preserves_fp8_and_e8m0_bytes(source, tmp_path, monkeypatch):
    _, inventory = read_inventory(source)
    names = ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale")
    destination = tmp_path / "preserved.safetensors"
    monkeypatch.setattr(CONVERTER, "COPY_BUFFER_BYTES", 5)
    copy_raw_tensors(destination, {name: inventory[name] for name in names})
    with safe_open(str(destination), framework="pt", device="cpu") as output:
        for name in names:
            info = inventory[name]
            with safe_open(str(info.path), framework="pt", device="cpu") as original:
                assert torch.equal(
                    output.get_tensor(name).view(torch.uint8), original.get_tensor(name).view(torch.uint8)
                )


@pytest.mark.parametrize("bits", [4, 8])
def test_complete_roundtrip_indices_config_and_tensor_coverage(source, tmp_path, bits):
    original_files = {path.name: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / f"converted_{bits}"
    report = convert_checkpoint(source, output, target_bits=bits, max_shard_bytes=2048)
    assert report["conversion_finished"] and report["complete_selection"]
    assert report["source_tensor_count"] == report["consumed_source_tensors"]
    assert report["projected_tensor_bytes"] == report["output_tensor_bytes"]
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == {"quant_method": "ascend"}
    assert "quantization_config" not in config["text_config"]
    assert config["text_config"]["dspark_block_size"] == 5
    assert config["vision_config"] == {"hidden_size": 32}
    assert (output / "tokenizer_config.json").is_file()
    assert not (output / "CONVERSION_INCOMPLETE").exists()
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index == json.loads((output / "quant_model_weights.safetensors.index.json").read_text())
    description = json.loads((output / "quant_model_description.json").read_text())
    actual_keys = set()
    for path in output.glob("*.safetensors"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            actual_keys.update(handle.keys())
    assert actual_keys == set(index["weight_map"])
    for name, filename in index["weight_map"].items():
        with safe_open(str(output / filename), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
            if name.endswith(".scale"):
                assert name == "layers.1.engram.embed.scale"
            if name.endswith(".weight") and ".engram.embed." not in name:
                assert tensor.dtype != torch.float8_e4m3fn
            if name.endswith(".w2.scale_bias"):
                assert tensor.shape[1] == 16
    assert description["mtp.0.main_proj.weight"] == FLOAT
    assert description["mtp.2.attn.wo_a.weight"] == FLOAT
    assert description["mtp.2.ffn.experts.0.w1.weight"] == W8A8
    assert description["layers.0.ffn.experts.0.w1.weight"] == (W4A8 if bits == 4 else W8A8)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original_files


def test_audit_does_not_create_output_and_partial_is_not_servable(source, tmp_path):
    output = tmp_path / "audit"
    report = convert_checkpoint(source, output, dry_run=True)
    assert report["dry_run"]
    assert not output.exists()
    partial = tmp_path / "partial"
    report = convert_checkpoint(source, partial, include=("layers.0.ffn.experts.0.*",))
    assert report["consumed_source_tensors"] == 6
    assert report["output_tensor_count"] == 12
    assert not report["complete_selection"]
    assert not (partial / "config.json").exists()
    assert (partial / "CONVERSION_INCOMPLETE").is_file()
    with pytest.raises(ValueError, match="empty directory"):
        convert_checkpoint(source, partial)
    with pytest.raises(ValueError, match="must not overlap"):
        convert_checkpoint(source, source / "nested")


def test_space_preflight_leaves_no_incomplete_output(source, tmp_path, monkeypatch):
    monkeypatch.setattr(CONVERTER.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    output = tmp_path / "not_enough_space"
    with pytest.raises(ValueError, match="Insufficient free disk"):
        convert_checkpoint(source, output)
    assert not output.exists()
