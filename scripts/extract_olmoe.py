#!/usr/bin/env python3
"""
extract_olmoe.py — Convert OLMoE-1B-7B safetensors → M1MoE binary format.

Usage:
    python extract_olmoe.py \
        --model-dir ~/models/OLMoE-1B-7B-0924-Instruct \
        --out-dir   ~/models/olmoe-m1moe

Outputs:
    <out-dir>/
        model_weights.bin      — flat non-expert weights (mmap-friendly)
        model_weights.json     — manifest {"name": {offset, shape, dtype}}
        packed_experts/
            layer_00.bin       — 64 experts × 4-bit packed per layer
            layer_01.bin
            ...
            layer_15.bin

Expert binary layout per file (layer_NN.bin):
    [expert_0][expert_1]...[expert_63]

Each expert block (gate_proj + up_proj + down_proj):
    gate_W  [intermediate × hidden/8]    uint32   packed nibbles (8 per u32)
    gate_S  [intermediate × hidden/gs]   uint16   BF16 scales
    gate_B  [intermediate × hidden/gs]   uint16   BF16 biases (min values)
    up_W    [intermediate × hidden/8]    uint32
    up_S    [intermediate × hidden/gs]   uint16
    up_B    [intermediate × hidden/gs]   uint16
    down_W  [hidden × intermediate/8]    uint32
    down_S  [hidden × intermediate/gs]   uint16
    down_B  [hidden × intermediate/gs]   uint16

Quantization: asymmetric 4-bit affine per group of groupSize columns.
    scale = (max - min) / 15
    bias  = min
    nibble = clamp(round((w - min) / scale), 0, 15)
    dequant: nibble * scale + bias

Non-expert weights are stored as BF16 (2 bytes per element), norm/scalar
weights as F32.  The manifest records exact offset + shape + dtype so the
Swift WeightStore can wrap them zero-copy.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ── safetensors loader (pure Python, no torch required) ─────────────────────

def load_safetensors_header(path):
    """Return (header_dict, data_offset) without loading tensor data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes)
    data_offset = 8 + header_len
    return header, data_offset


def iter_safetensors(path, keys=None):
    """
    Yield (name, numpy_array) for all tensors in a safetensors file.
    If keys is not None, yield only those keys.
    Supports dtypes: BF16 (returned as uint16), F16, F32.
    """
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
    """Convert uint16 BF16 array to float32 numpy array."""
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(arr_f32):
    """Convert float32 numpy array to uint16 BF16 (round-to-nearest-even)."""
    arr = arr_f32.astype(np.float32)
    # Round: add 0x7FFF + bit15 of the original mantissa
    u32 = arr.view(np.uint32)
    rounding_bias = (u32 >> 16) & 1
    rounded = u32 + 0x7FFF + rounding_bias
    return (rounded >> 16).astype(np.uint16)


# ── 4-bit group quantization ─────────────────────────────────────────────────

def quantize_4bit(weight_f32, group_size=64):
    """
    Asymmetric 4-bit affine quantization over groups of columns.

    Input:  weight_f32 — float32 array [rows, cols]
    Output: (packed_u32, scales_bf16, biases_bf16)
        packed_u32  — uint32 [rows, cols/8]   — 8 nibbles per word
        scales_bf16 — uint16 [rows, cols/gs]  — BF16 scale per group
        biases_bf16 — uint16 [rows, cols/gs]  — BF16 bias  per group

    Dequant: w ≈ nibble * scale + bias
    """
    rows, cols = weight_f32.shape
    assert cols % group_size == 0, f"cols {cols} not divisible by group_size {group_size}"
    assert cols % 8 == 0, f"cols {cols} not divisible by 8"

    num_groups = cols // group_size
    w = weight_f32.reshape(rows, num_groups, group_size)  # [R, G, gs]

    w_min = w.min(axis=2, keepdims=True)   # [R, G, 1]
    w_max = w.max(axis=2, keepdims=True)   # [R, G, 1]

    # Avoid zero scale for constant-zero groups
    scale = (w_max - w_min) / 15.0
    scale = np.where(scale == 0, np.ones_like(scale), scale)

    # Quantize
    q = np.clip(np.round((w - w_min) / scale), 0, 15).astype(np.uint8)  # [R, G, gs]

    # Pack 8 nibbles per uint32 (little-endian nibble order: nibble0 in bits 0-3)
    q_flat = q.reshape(rows, cols)            # [R, C]
    q_grouped_8 = q_flat.reshape(rows, cols // 8, 8)  # [R, C/8, 8]
    packed = np.zeros((rows, cols // 8), dtype=np.uint32)
    for b in range(8):
        packed |= q_grouped_8[:, :, b].astype(np.uint32) << (b * 4)

    # Scales and biases — convert to BF16
    scales_f32 = scale.reshape(rows, num_groups).astype(np.float32)
    biases_f32 = w_min.reshape(rows, num_groups).astype(np.float32)

    scales_bf16 = f32_to_bf16(scales_f32)
    biases_bf16 = f32_to_bf16(biases_f32)

    return packed, scales_bf16, biases_bf16


def expert_bytes(hidden_dim, intermediate_dim, group_size):
    """Compute expected byte size for one packed expert (must match ModelSpec.expertBytes4Bit)."""
    h, i, gs = hidden_dim, intermediate_dim, group_size
    gw  = i * (h // 8) * 4
    gsb = i * (h // gs) * 2
    dw  = h * (i // 8) * 4
    dsb = h * (i // gs) * 2
    return (gw + gsb * 2) + (gw + gsb * 2) + (dw + dsb * 2)


# ── Main extraction logic ─────────────────────────────────────────────────────

def find_safetensors_shards(model_dir):
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return shards


def load_all_tensor_names(shards):
    """Return {name: shard_path} for every tensor across all shards."""
    mapping = {}
    for shard in shards:
        header, _ = load_safetensors_header(shard)
        for name in header:
            if name != "__metadata__":
                mapping[name] = shard
    return mapping


def is_expert_weight(name):
    """True if this tensor belongs to an MoE expert (not the router gate)."""
    # OLMoE expert naming: model.layers.{i}.mlp.experts.{j}.{proj}.weight
    return ".mlp.experts." in name


def extract(model_dir, out_dir, group_size=64, verbose=True):
    model_dir = Path(model_dir).expanduser()
    out_dir   = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    experts_dir = out_dir / "packed_experts"
    experts_dir.mkdir(exist_ok=True)

    shards = find_safetensors_shards(model_dir)
    tensor_map = load_all_tensor_names(shards)

    # Split names into non-expert and expert
    non_expert_names = sorted([n for n in tensor_map if not is_expert_weight(n)])
    expert_names     = sorted([n for n in tensor_map if is_expert_weight(n)])

    if verbose:
        print(f"[extract] {len(non_expert_names)} non-expert tensors")
        print(f"[extract] {len(expert_names)} expert tensors")

    # ── 1. Non-expert weights → model_weights.bin ────────────────────────────

    bin_path  = out_dir / "model_weights.bin"
    json_path = out_dir / "model_weights.json"

    manifest = {}
    offset   = 0

    # Group names by shard to minimise file seeks
    shard_to_names = {}
    for name in non_expert_names:
        shard = tensor_map[name]
        shard_to_names.setdefault(str(shard), []).append(name)

    with open(bin_path, "wb") as bin_f:
        for shard_path_str, names in shard_to_names.items():
            if verbose:
                print(f"  [non-expert] loading {Path(shard_path_str).name} …")
            for name, arr in iter_safetensors(shard_path_str, keys=set(names)):
                # Convert to the storage dtype
                if arr.dtype == np.uint16:
                    # Assumed BF16 — keep as-is
                    dtype = "bf16"
                    raw = arr.tobytes()
                elif arr.dtype == np.float16:
                    # Upcast F16 → F32 → BF16 to avoid precision loss surprises
                    f32 = arr.astype(np.float32)
                    bf16 = f32_to_bf16(f32)
                    dtype = "bf16"
                    raw = bf16.tobytes()
                else:
                    # F32 — keep as-is (norms, etc.)
                    dtype = "f32"
                    raw = arr.astype(np.float32).tobytes()

                # Align to 4 bytes
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

    # Parse expert names into {(layer, expert): {proj: shard_path}}
    # OLMoE naming: "model.layers.{L}.mlp.experts.{E}.{proj}.weight"
    from collections import defaultdict
    expert_dict = defaultdict(dict)  # (layer, expert) → {proj: shard_path}

    for name in expert_names:
        parts = name.split(".")
        # model.layers.L.mlp.experts.E.proj.weight  → indices [2]=L, [5]=E, [6]=proj
        try:
            L    = int(parts[2])
            E    = int(parts[5])
            proj = parts[6]   # "gate_proj" | "up_proj" | "down_proj"
        except (IndexError, ValueError):
            print(f"  [warn] unexpected expert tensor name: {name}", file=sys.stderr)
            continue
        expert_dict[(L, E)][proj] = (name, tensor_map[name])

    layers = sorted(set(L for L, _ in expert_dict.keys()))

    # Detect dims from first expert
    first_key = min(expert_dict.keys())
    gate_name, gate_shard = expert_dict[first_key]["gate_proj"]
    tmp_header, _ = load_safetensors_header(gate_shard)
    gate_shape = tmp_header[gate_name]["shape"]   # [intermediate, hidden]
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

        # Load all weight tensors for this layer in one pass per shard
        # Collect which shard has which name
        layer_shard_names = {}   # shard_path → set of names
        for E in experts_in_layer:
            for proj, (name, shard) in expert_dict[(L, E)].items():
                layer_shard_names.setdefault(str(shard), set()).add(name)

        loaded = {}   # name → arr
        for shard_path_str, names in layer_shard_names.items():
            for name, arr in iter_safetensors(shard_path_str, keys=names):
                loaded[name] = arr

        with open(layer_path, "wb") as lf:
            for E in experts_in_layer:
                projs = expert_dict[(L, E)]
                for proj_key in ("gate_proj", "up_proj", "down_proj"):
                    name, _ = projs[proj_key]
                    arr = loaded[name]
                    # Convert to F32 for quantization
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
        total_mb = sum(
            os.path.getsize(experts_dir / f"layer_{L:02d}.bin") for L in layers
        ) / 1_048_576
        print(f"[extract] packed_experts/: {total_mb:.0f} MB total")
        print("[extract] done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Extract OLMoE weights to M1MoE format")
    p.add_argument("--model-dir", required=True,
                   help="Path to OLMoE-1B-7B-0924-Instruct directory (contains *.safetensors)")
    p.add_argument("--out-dir",   required=True,
                   help="Output directory (created if needed)")
    p.add_argument("--group-size", type=int, default=64,
                   help="Quantization group size (default: 64)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    extract(args.model_dir, args.out_dir,
            group_size=args.group_size,
            verbose=not args.quiet)


if __name__ == "__main__":
    main()
