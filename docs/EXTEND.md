# Adding Models to M1MoE

This guide explains how to download a Mixture-of-Experts model from HuggingFace and convert it into the binary format M1MoE requires.

---

## Overview of the Pipeline

```
HuggingFace repo
  (*.safetensors shards + tokenizer.json)
        │
        │  extract_*.py
        ▼
  model_weights.bin       ← non-expert weights, BF16/F32, mmap-ready
  model_weights.json      ← tensor manifest (name → offset/shape/dtype)
  packed_experts/
    layer_00.bin          ← all experts for layer 0, 4-bit packed
    layer_01.bin
    …
  tokenizer.json          ← copied as-is from the HF repo
        │
        │  configs/*.json
        ▼
  M1MoE runtime
```

The extraction scripts are pure Python — they read `.safetensors` files directly (no PyTorch required) and write the above layout. The Swift engine then `mmap`s `model_weights.bin` and `pread`s expert blocks on demand.

---

## Prerequisites

```bash
# Python 3.9+
pip install numpy huggingface_hub
```

`numpy` is the only runtime dependency of the extraction scripts.
`huggingface_hub` provides the `huggingface-cli` download tool.

To download gated models (Mixtral, Llama, etc.) you need a HuggingFace account and must accept the model's license on the model card page before downloading.

```bash
huggingface-cli login          # paste your HF access token once
```

---

## Supported Models (Ready to Use)

### OLMoE-1B-7B-0924-Instruct

Allen AI's sparse MoE model. 16 layers, 64 experts per layer, top-8 routing, 2048 hidden dim. ~2.5 GB download; ~1.4 GB converted.

**1. Download**

```bash
huggingface-cli download \
  allenai/OLMoE-1B-7B-0924-Instruct \
  --local-dir ~/models/OLMoE-1B-7B-0924-Instruct \
  --include "*.safetensors" "tokenizer.json"
```

**2. Extract**

```bash
python scripts/extract_olmoe.py \
  --model-dir ~/models/OLMoE-1B-7B-0924-Instruct \
  --out-dir   ~/models/olmoe-m1moe
```

Runtime: ~5–10 minutes on an M-series Mac (quantization is the bottleneck).

**3. Copy tokenizer**

The extract script does not copy the tokenizer. Do it manually:

```bash
cp ~/models/OLMoE-1B-7B-0924-Instruct/tokenizer.json \
   ~/models/olmoe-m1moe/tokenizer.json
```

**4. Config** — already provided at `configs/olmoe-1b-7b.json`. Update `model_dir`:

```json
{
  "name": "OLMoE-1B-7B-0924",
  "architecture": "olmoe",
  "model_dir": "/Users/YOUR_NAME/models/olmoe-m1moe",
  ...
}
```

**5. Run**

```bash
swift run -c release chat --config configs/olmoe-1b-7b.json
```

---

### Mixtral-8x7B-Instruct-v0.1

Mistral AI's sparse MoE. 32 layers, 8 experts per layer, top-2 routing, 4096 hidden dim. ~90 GB download (8-shard safetensors); ~26 GB converted. Requires accepting the license on HuggingFace.

**1. Download**

```bash
huggingface-cli download \
  mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir ~/models/Mixtral-8x7B-Instruct-v0.1 \
  --include "*.safetensors" "tokenizer.json" "tokenizer.model"
```

This downloads ~90 GB across 8 shards. Allow several hours on a typical home connection.

**2. Extract**

```bash
python scripts/extract_mixtral.py \
  --model-dir ~/models/Mixtral-8x7B-Instruct-v0.1 \
  --out-dir   ~/models/mixtral-m1moe
```

Runtime: 45–90 minutes (32 layers × 8 experts, each 14336-dim).
Peak RAM during extraction: ~20 GB (one full layer loaded at a time).

**3. Copy tokenizer**

```bash
cp ~/models/Mixtral-8x7B-Instruct-v0.1/tokenizer.json \
   ~/models/mixtral-m1moe/tokenizer.json
```

**4. Config** — already provided at `configs/mixtral-8x7b.json`. Update `model_dir`.

**5. Run**

```bash
swift run -c release chat --config configs/mixtral-8x7b.json
```

Mixtral requires at least 32 GB of system RAM to hold the non-expert weights plus KV cache. The expert window fills the remainder.

---

## Adding a New Model

M1MoE can support any decoder-only MoE model that shares the same basic transformer structure: RMS norm → multi-head attention with RoPE → MoE FFN block (router + top-K experts with SwiGLU). Adding a new model involves three steps: writing an extraction script, writing a JSON config, and (if the model uses a different attention variant) possibly a small engine change.

### Step 1 — Understand What the Engine Expects

**Non-expert tensor names** must exactly match these keys (used in `InferenceEngine.swift` via `WeightKeys`):

| Tensor | Name in model_weights.json |
|---|---|
| Token embeddings | `model.embed_tokens.weight` |
| Final RMS norm | `model.norm.weight` |
| LM head | `lm_head.weight` |
| Input layer norm (each layer L) | `model.layers.L.input_layernorm.weight` |
| Post-attention norm (each layer L) | `model.layers.L.post_attention_layernorm.weight` |
| Q projection | `model.layers.L.self_attn.q_proj.weight` |
| K projection | `model.layers.L.self_attn.k_proj.weight` |
| V projection | `model.layers.L.self_attn.v_proj.weight` |
| O projection | `model.layers.L.self_attn.o_proj.weight` |
| Q head norm _(OLMoE only)_ | `model.layers.L.self_attn.q_norm.weight` |
| K head norm _(OLMoE only)_ | `model.layers.L.self_attn.k_norm.weight` |
| MoE router (olmoe arch) | `model.layers.L.mlp.gate.weight` |
| MoE router (mixtral arch) | `model.layers.L.block_sparse_moe.gate.weight` |

If your model uses the same tensor names as one of these architectures (common for LLaMA-derivative MoE models), you can reuse the existing architecture string. If the names differ, you either rename them during extraction or add a new architecture string and a new case in `WeightKeys.router()` in `InferenceEngine.swift`.

**Expert binary layout** inside each `packed_experts/layer_NN.bin`:

```
[expert_0 block][expert_1 block]…[expert_E-1 block]

Each block, in order:
  gate_proj packed   u32  [intermediateDim × (hiddenDim/8)]
  gate_proj scales   u16  [intermediateDim × (hiddenDim/groupSize)]  (BF16)
  gate_proj biases   u16  [intermediateDim × (hiddenDim/groupSize)]  (BF16)
  up_proj   packed   u32  [intermediateDim × (hiddenDim/8)]
  up_proj   scales   u16  [intermediateDim × (hiddenDim/groupSize)]
  up_proj   biases   u16  [intermediateDim × (hiddenDim/groupSize)]
  down_proj packed   u32  [hiddenDim × (intermediateDim/8)]
  down_proj scales   u16  [hiddenDim × (intermediateDim/groupSize)]
  down_proj biases   u16  [hiddenDim × (intermediateDim/groupSize)]
```

The order gate → up → down is hardcoded in both the extraction scripts and the `InferenceEngine` via its `gateWOff`/`upWOff`/`downWOff` byte-offset constants. Do not change this order.

**Dimension constraints:**

```
hiddenDim       % 8          == 0   (8 nibbles packed per uint32)
hiddenDim       % groupSize  == 0   (groupSize columns share one scale/bias)
intermediateDim % 8          == 0
intermediateDim % groupSize  == 0
```

With the default `groupSize=64` these are satisfied by all common hidden dims (2048, 4096, 7168, 14336, etc.).

### Step 2 — Write the Extraction Script

Copy `scripts/extract_olmoe.py` or `scripts/extract_mixtral.py` as a starting point. The only parts you need to change are:

**a) `is_expert_weight(name)`** — return `True` for tensors that belong to experts:

```python
def is_expert_weight(name):
    # Adjust the substring to match your model's naming scheme.
    # OLMoE: ".mlp.experts."
    # Mixtral: ".block_sparse_moe.experts."
    # Example for a hypothetical model:
    return ".moe_block.experts." in name
```

**b) The expert name parser** — extract `(layer, expert_index, projection)` from each tensor name. OLMoE uses:

```
model.layers.{L}.mlp.experts.{E}.gate_proj.weight
```

Mixtral uses `w1`/`w2`/`w3` instead of `gate_proj`/`down_proj`/`up_proj`, so the script includes a rename map:

```python
MIXTRAL_PROJ_MAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
```

For your model, adjust the index positions in `parts = name.split(".")` and the projection key mapping as needed.

**c) Non-expert tensor renaming** — if your model's attention weight names differ from the expected names above, add a rename step when writing the manifest:

```python
# Example: your model uses "model.layers.L.attention.wq" instead of
# "model.layers.L.self_attn.q_proj.weight"
RENAME = {
    "attention.wq":    "self_attn.q_proj.weight",
    "attention.wk":    "self_attn.k_proj.weight",
    "attention.wv":    "self_attn.v_proj.weight",
    "attention.wo":    "self_attn.o_proj.weight",
    "feed_forward.gate": "mlp.gate.weight",
}

def canonical_name(name):
    for old, new in RENAME.items():
        if old in name:
            return name.replace(old, new)
    return name

# Then use canonical_name(name) when building manifest[...]
```

**d) Router weight name** — if your architecture uses a different router key, either rename it to match an existing architecture string's expected key, or add a new case in `WeightKeys.router()` in `InferenceEngine.swift`.

Everything else — the quantization function, the binary packing loop, the manifest format, the shard loading — is identical across all models and can be reused unchanged.

### Step 3 — Write the JSON Config

Create `configs/your-model.json`. Every field is required unless marked optional.

```jsonc
{
  // ── Identity ────────────────────────────────────────────────────────────────
  "name": "YourModel-Name",

  // "olmoe" activates QK head norms and uses mlp.gate.weight router key.
  // "mixtral" skips QK norms and uses block_sparse_moe.gate.weight.
  // Use whichever matches the tensor names you wrote in the extraction script.
  "architecture": "olmoe",

  "model_dir": "/Users/you/models/your-model-m1moe",

  // ── Core dimensions ─────────────────────────────────────────────────────────
  "hidden_dim": 2048,       // d_model
  "vocab_size": 50304,      // must match embed_tokens.weight shape[0]
  "num_layers": 16,
  "rms_norm_eps": 1e-5,     // check config.json in HF repo; often 1e-5 or 1e-6

  // ── Attention ───────────────────────────────────────────────────────────────
  "attention": {
    "num_heads": 16,         // Q heads
    "num_kv_heads": 16,      // KV heads (= num_heads for MHA; < for GQA)
    "head_dim": 128,         // hidden_dim / num_heads for standard models
    "rope_theta": 10000.0,   // RoPE base frequency; check model's config.json
    "max_seq_len": 4096,     // maximum sequence length the model was trained on
    "gpu_kv_seq_prealloc": 4096  // Metal KV cache pre-alloc; can be <= max_seq_len
  },

  // ── Mixture of Experts ──────────────────────────────────────────────────────
  "moe": {
    "num_experts": 64,       // total experts per layer
    "top_k": 8,              // how many experts are activated per token
    "intermediate_dim": 1024,// FFN hidden dim inside each expert
    "norm_topk_prob": false  // true = normalize routing weights to sum=1
  },

  // ── Quantization ────────────────────────────────────────────────────────────
  // Must match --group-size passed to the extraction script.
  "quantization": {
    "bits": 4,
    "group_size": 64
  },

  // ── Special tokens ──────────────────────────────────────────────────────────
  // Find these in tokenizer_config.json or tokenizer.json in the HF repo.
  "special_tokens": {
    "eos": [50279],   // array — some models have multiple EOS tokens
    "bos": 1
  },

  // ── Runtime ─────────────────────────────────────────────────────────────────
  // expert_window_size is overridden at runtime by HardwareProfile if RAM allows.
  // Set it to a sane fallback (e.g. num_layers × top_k for a "perfect" window).
  "runtime": {
    "expert_window_size": 128,
    "io_threads": 8
  },

  // ── Chat template (optional) ─────────────────────────────────────────────────
  // Omit this section to use the architecture default (see ModelSpec.chatTemplate).
  // Include it to override with your model's exact format.
  "chat": {
    "user_prefix":       "<|user|>\n",
    "user_suffix":       "\n<|assistant|>\n",
    "assistant_prefix":  "",
    "assistant_suffix":  "<|endoftext|>\n",
    "system_prefix":     "<|system|>\n",
    "system_suffix":     "\n"
  }
}
```

**Where to find these values** for any HuggingFace model:

| Config field | Source in HF repo |
|---|---|
| `hidden_dim` | `config.json` → `hidden_size` |
| `vocab_size` | `config.json` → `vocab_size` |
| `num_layers` | `config.json` → `num_hidden_layers` |
| `rms_norm_eps` | `config.json` → `rms_norm_eps` |
| `num_heads` | `config.json` → `num_attention_heads` |
| `num_kv_heads` | `config.json` → `num_key_value_heads` |
| `head_dim` | `config.json` → `head_dim` or `hidden_size / num_attention_heads` |
| `rope_theta` | `config.json` → `rope_theta` |
| `max_seq_len` | `config.json` → `max_position_embeddings` |
| `num_experts` | `config.json` → `num_experts` or `num_local_experts` |
| `top_k` | `config.json` → `num_experts_per_tok` |
| `intermediate_dim` | `config.json` → `intermediate_size` (inside `ffn_config` for some models) |
| `norm_topk_prob` | `config.json` → `norm_topk_prob` (default false) |
| EOS token ID | `tokenizer_config.json` → `eos_token_id` |
| BOS token ID | `tokenizer_config.json` → `bos_token_id` |
| Chat template | `tokenizer_config.json` → `chat_template` (Jinja; translate manually) |

### Step 4 — Verify the Extraction

After running your extraction script, check:

```bash
# 1. Layer files exist and are the right size
ls -lh ~/models/your-model-m1moe/packed_experts/

# Expected: numLayers files, each exactly numExperts × expertBytes bytes.
# The extraction script prints the expected size and will exit(1) on mismatch.

# 2. Manifest covers all needed tensors
python3 - <<'EOF'
import json
m = json.load(open("model_weights.json"))
needed = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate.weight",   # or block_sparse_moe.gate.weight
]
for n in needed:
    print("OK" if n in m else "MISSING", n)
EOF

# 3. Quick sanity run with bench
swift run -c release bench \
  --config configs/your-model.json \
  --tokens 5 --temp 0
```

The bench run will immediately crash with a clear error message if any required tensor is missing from the manifest.

---

## Models Compatible with Existing Scripts

The following HuggingFace models are structurally identical to OLMoE or Mixtral and can be converted with the existing scripts. You only need to write a new config JSON.

### Mixtral-8x22B-Instruct-v0.1

Same architecture as Mixtral-8x7B. Use `extract_mixtral.py` unchanged.

```
huggingface-cli download mistralai/Mixtral-8x22B-Instruct-v0.1 \
  --local-dir ~/models/Mixtral-8x22B-Instruct-v0.1 \
  --include "*.safetensors" "tokenizer.json"
```

Config differences from `mixtral-8x7b.json`:

```json
{
  "name": "Mixtral-8x22B-Instruct-v0.1",
  "architecture": "mixtral",
  "hidden_dim": 6144,
  "vocab_size": 32768,
  "num_layers": 56,
  "attention": {
    "num_heads": 48,
    "num_kv_heads": 8,
    "head_dim": 128,
    "rope_theta": 1000000.0,
    "max_seq_len": 65536,
    "gpu_kv_seq_prealloc": 8192
  },
  "moe": {
    "num_experts": 8,
    "top_k": 2,
    "intermediate_dim": 16384,
    "norm_topk_prob": false
  }
}
```

**Requirements:** ~280 GB disk for download, ~75 GB converted. Needs 96+ GB RAM to run (M2 Ultra/M3 Ultra/M4 Ultra class machine).

---

## Architecture Constraints and Limitations

The following model types are **not directly supported** without changes to the Swift engine:

| Feature | Limitation |
|---|---|
| **Shared experts** (DeepSeek-V2/V3) | The engine has no concept of always-active experts separate from routed experts. Would need a new forward-pass path. |
| **Dense FFN layers** (mixed dense+MoE models) | The engine assumes every layer has a router + top-K experts. A layer with a standard dense FFN would need a code path that skips routing. |
| **ALiBi / non-RoPE positional encodings** | RoPE is hardcoded in both CPU (`applyRoPE`) and GPU (`ropeInPlace` kernel). |
| **Sliding window attention** | The KV cache stores all positions up to `gpuKvSeqPrealloc`. No per-layer window masking is implemented. |
| **Tied embeddings** (`lm_head` shares weights with `embed_tokens`) | The manifest must contain a separate `lm_head.weight` entry. If the model ties these, copy the embedding tensor under the `lm_head.weight` key in the extraction script. |
| **Multiple norm variants** (LayerNorm, QKNorm per architecture) | QK head norms are enabled only for the `olmoe` architecture string. Any other model needing QK norms would need `olmoe` architecture or a new string + `hasQKNorm` flag in `InferenceEngine`. |
| **INT8 or FP8 quantization** | Only 4-bit group quantization with `groupSize=64` is implemented. The config `bits` field has no other paths in the engine. |

---

## Disk Space and Time Reference

| Model | HF Download | Converted Size | Extraction Time (M2 Max) |
|---|---|---|---|
| OLMoE-1B-7B | ~2.5 GB | ~1.4 GB | ~8 min |
| Mixtral-8x7B | ~90 GB | ~26 GB | ~75 min |
| Mixtral-8x22B | ~280 GB | ~75 GB | ~4 hrs |

Conversion is CPU/RAM-bound (numpy quantization), not I/O-bound. Peak RAM during extraction equals roughly one full layer of expert weights in F32, which is:

```
peak_RAM ≈ numExperts × intermediateDim × hiddenDim × 3 × 4 bytes
         = for Mixtral: 8 × 14336 × 4096 × 3 × 4 ≈ 5.6 GB per layer
```

The extraction script processes one layer at a time, so this is the peak, not the total.

---

## Quick Reference: Differences Between the Two Existing Scripts

| Aspect | extract_olmoe.py | extract_mixtral.py |
|---|---|---|
| Expert tensor marker | `.mlp.experts.` | `.block_sparse_moe.experts.` |
| Expert name format | `…experts.{E}.gate_proj.weight` | `…experts.{E}.w1` / `w2` / `w3` |
| Projection rename | None needed | `w1→gate_proj`, `w3→up_proj`, `w2→down_proj` |
| Router tensor key | `model.layers.L.mlp.gate.weight` | `model.layers.L.block_sparse_moe.gate.weight` |
| Everything else | Identical | Identical |

The quantization format, binary packing, non-expert extraction, and manifest structure are byte-for-byte identical between the two scripts. Any new extraction script for a similarly structured model can copy those sections verbatim.
