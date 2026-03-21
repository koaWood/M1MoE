#!/usr/bin/env python3
"""
extract_mixtral.py — Convert Mixtral-8x7B safetensors → M1MoE binary format.

Usage:
    python extract_mixtral.py \
        --model-dir ~/models/Mixtral-8x7B-Instruct-v0.1 \
        --out-dir   ~/models/mixtral-m1moe

Outputs:
    <out-dir>/
        model_weights.bin      — flat non-expert weights (mmap-friendly)
        model_weights.json     — manifest {"name": {offset, shape, dtype}}
        packed_experts/
            layer_00.bin       — 8 experts × 4-bit packed per layer
            layer_01.bin
            ...
            layer_31.bin

Mixtral expert naming:
    model.layers.{L}.block_sparse_moe.experts.{E}.w1  — gate_proj [intermediate, hidden]
    model.layers.{L}.block_sparse_moe.experts.{E}.w3  — up_proj   [intermediate, hidden]
    model.layers.{L}.block_sparse_moe.experts.{E}.w2  — down_proj [hidden, intermediate]

Router:
    model.layers.{L}.block_sparse_moe.gate.weight  — [numExperts, hidden]

Expert binary layout and quantization: identical to extract_olmoe.py.
See that file for format documentation.
"""

import argparse
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Shared helpers (identical to extract_olmoe.py) ───────────────────────────

def load_safetensors_header(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes)
    data_offset = 8 + header_len
    return header, data_offset


def iter_safetensors(path, keys=None):
    header, data_offset = load_safetensors_header(path)
    with open(path, "rb") as f:
        f.seek(data_offset)
        data = f.read()

    for name, info in header.items():
        if name == "__metadata__":
            continue
        if keys is not None and name not in keys:
            continue
        dtype_str = info["dtype"]
        shape     = info["shape"]
        start, end = info["data_offsets"]
        raw = data[start:end]
        if dtype_str == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        elif dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        else:
            print(f"  [skip] {name}: unsupported dtype {dtype_str}", file=sys.stderr)
            continue
        yield name, arr


def bf16_to_f32(arr_u16):
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(arr_f32):
    arr = arr_f32.astype(np.float32)
    u32 = arr.view(np.uint32)
    rounding_bias = (u32 >> 16) & 1
    rounded = u32 + 0x7FFF + rounding_bias
    return (rounded >> 16).astype(np.uint16)


def quantize_4bit(weight_f32, group_size=64):
    rows, cols = weight_f32.shape
    assert cols % group_size == 0
    assert cols % 8 == 0

    num_groups = cols // group_size
    w = weight_f32.reshape(rows, num_groups, group_size)

    w_min = w.min(axis=2, keepdims=True)
    w_max = w.max(axis=2, keepdims=True)
    scale = (w_max - w_min) / 15.0
    scale = np.where(scale == 0, np.ones_like(scale), scale)

    q = np.clip(np.round((w - w_min) / scale), 0, 15).astype(np.uint8)
    q_flat = q.reshape(rows, cols)
    q_grouped_8 = q_flat.reshape(rows, cols // 8, 8)
    packed = np.zeros((rows, cols // 8), dtype=np.uint32)
    for b in range(8):
        packed |= q_grouped_8[:, :, b].astype(np.uint32) << (b * 4)

    scales_bf16 = f32_to_bf16(scale.reshape(rows, num_groups).astype(np.float32))
    biases_bf16 = f32_to_bf16(w_min.reshape(rows, num_groups).astype(np.float32))
    return packed, scales_bf16, biases_bf16


def expert_bytes(hidden_dim, intermediate_dim, group_size):
    h, i, gs = hidden_dim, intermediate_dim, group_size
    gw  = i * (h // 8) * 4
    gsb = i * (h // gs) * 2
    dw  = h * (i // 8) * 4
    dsb = h * (i // gs) * 2
    return (gw + gsb * 2) + (gw + gsb * 2) + (dw + dsb * 2)


def find_safetensors_shards(model_dir):
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return shards


def load_all_tensor_names(shards):
    mapping = {}
    for shard in shards:
        header, _ = load_safetensors_header(shard)
        for name in header:
            if name != "__metadata__":
                mapping[name] = shard
    return mapping


def is_expert_weight(name):
    # Mixtral: model.layers.{L}.block_sparse_moe.experts.{E}.{w1|w2|w3}
    return ".block_sparse_moe.experts." in name


# ── Main ─────────────────────────────────────────────────────────────────────

def extract(model_dir, out_dir, group_size=64, verbose=True):
    model_dir = Path(model_dir).expanduser()
    out_dir   = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    experts_dir = out_dir / "packed_experts"
    experts_dir.mkdir(exist_ok=True)

    shards = find_safetensors_shards(model_dir)
    tensor_map = load_all_tensor_names(shards)

    non_expert_names = sorted([n for n in tensor_map if not is_expert_weight(n)])
    expert_names     = sorted([n for n in tensor_map if is_expert_weight(n)])

    if verbose:
        print(f"[extract] {len(non_expert_names)} non-expert tensors")
        print(f"[extract] {len(expert_names)} expert weight tensors")

    # ── 1. Non-expert weights → model_weights.bin ────────────────────────────

    bin_path  = out_dir / "model_weights.bin"
    json_path = out_dir / "model_weights.json"

    manifest = {}
    offset   = 0

    shard_to_names = {}
    for name in non_expert_names:
        shard = tensor_map[name]
        shard_to_names.setdefault(str(shard), []).append(name)

    with open(bin_path, "wb") as bin_f:
        for shard_path_str, names in shard_to_names.items():
            if verbose:
                print(f"  [non-expert] {Path(shard_path_str).name} …")
            for name, arr in iter_safetensors(shard_path_str, keys=set(names)):
                if arr.dtype == np.uint16:
                    dtype = "bf16"
                    raw = arr.tobytes()
                elif arr.dtype == np.float16:
                    f32 = arr.astype(np.float32)
                    bf16 = f32_to_bf16(f32)
                    dtype = "bf16"
                    raw = bf16.tobytes()
                else:
                    dtype = "f32"
                    raw = arr.astype(np.float32).tobytes()

                if offset % 4 != 0:
                    pad = 4 - (offset % 4)
                    bin_f.write(b"\x00" * pad)
                    offset += pad

                manifest[name] = {"offset": offset, "shape": list(arr.shape), "dtype": dtype}
                bin_f.write(raw)
                offset += len(raw)

    with open(json_path, "w") as jf:
        json.dump(manifest, jf, indent=2)

    if verbose:
        print(f"[extract] model_weights.bin: {offset / 1_048_576:.1f} MB")
        print(f"[extract] model_weights.json: {len(manifest)} entries")

    # ── 2. Expert weights → packed_experts/layer_NN.bin ──────────────────────

    # Parse Mixtral expert names:
    # model.layers.{L}.block_sparse_moe.experts.{E}.{w1|w2|w3}
    # w1 = gate_proj, w3 = up_proj, w2 = down_proj
    MIXTRAL_PROJ_MAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

    expert_dict = defaultdict(dict)   # (layer, expert) → {proj_key: (name, shard)}
    for name in expert_names:
        parts = name.split(".")
        # model.layers.L.block_sparse_moe.experts.E.wN
        # idx:  0     1  2  3                4       5  6
        try:
            L      = int(parts[2])
            E      = int(parts[5])
            w_key  = parts[6]   # "w1" | "w2" | "w3"
            proj   = MIXTRAL_PROJ_MAP.get(w_key)
        except (IndexError, ValueError):
            print(f"  [warn] unexpected: {name}", file=sys.stderr)
            continue
        if proj is None:
            continue
        expert_dict[(L, E)][proj] = (name, tensor_map[name])

    layers = sorted(set(L for L, _ in expert_dict.keys()))

    # Detect dims from first expert's gate_proj
    first_key = min(expert_dict.keys())
    gate_name, gate_shard = expert_dict[first_key]["gate_proj"]
    tmp_header, _ = load_safetensors_header(gate_shard)
    gate_shape = tmp_header[gate_name]["shape"]
    intermediate_dim, hidden_dim = gate_shape[0], gate_shape[1]

    exp_bytes = expert_bytes(hidden_dim, intermediate_dim, group_size)
    if verbose:
        print(f"[extract] hidden={hidden_dim}, intermediate={intermediate_dim}, gs={group_size}")
        print(f"[extract] expert size: {exp_bytes / 1_048_576:.2f} MB each")

    for L in layers:
        experts_in_layer = sorted(E for (l, E) in expert_dict.keys() if l == L)
        num_experts = len(experts_in_layer)
        layer_path = experts_dir / f"layer_{L:02d}.bin"

        if verbose:
            print(f"  [layer {L:02d}] {num_experts} experts …", end="", flush=True)

        # Load all tensors for this layer
        layer_shard_names = {}
        for E in experts_in_layer:
            for proj, (name, shard) in expert_dict[(L, E)].items():
                layer_shard_names.setdefault(str(shard), set()).add(name)

        loaded = {}
        for shard_path_str, names in layer_shard_names.items():
            for name, arr in iter_safetensors(shard_path_str, keys=names):
                loaded[name] = arr

        with open(layer_path, "wb") as lf:
            for E in experts_in_layer:
                projs = expert_dict[(L, E)]
                for proj_key in ("gate_proj", "up_proj", "down_proj"):
                    name, _ = projs[proj_key]
                    arr = loaded[name]
                    if arr.dtype == np.uint16:
                        w_f32 = bf16_to_f32(arr)
                    elif arr.dtype == np.float16:
                        w_f32 = arr.astype(np.float32)
                    else:
                        w_f32 = arr.astype(np.float32)

                    packed, scales, biases = quantize_4bit(w_f32, group_size)
                    lf.write(packed.tobytes())
                    lf.write(scales.tobytes())
                    lf.write(biases.tobytes())

        actual = os.path.getsize(layer_path)
        expected = num_experts * exp_bytes
        if actual != expected:
            print(f"\n  [ERROR] layer {L:02d}: {actual} bytes, expected {expected}")
            sys.exit(1)

        if verbose:
            print(f" {actual / 1_048_576:.0f} MB")

    if verbose:
        total = sum(
            os.path.getsize(experts_dir / f"layer_{L:02d}.bin") for L in layers
        ) / 1_048_576
        print(f"[extract] packed_experts/: {total:.0f} MB total ({total/1024:.1f} GB)")
        print("[extract] done.")


def main():
    p = argparse.ArgumentParser(description="Extract Mixtral-8x7B weights to M1MoE format")
    p.add_argument("--model-dir", required=True,
                   help="Path to Mixtral-8x7B-Instruct directory")
    p.add_argument("--out-dir",   required=True,
                   help="Output directory")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    extract(args.model_dir, args.out_dir,
            group_size=args.group_size,
            verbose=not args.quiet)


if __name__ == "__main__":
    main()
