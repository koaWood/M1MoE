// Shaders.metal — M1MoE compute kernels
//
// All kernels are parameterized by runtime constants passed via setBytes.
// No architecture-specific constants are compiled in.

#include <metal_stdlib>
using namespace metal;

// ── Helpers ─────────────────────────────────────────────────────────────────

inline float bf16_to_f32(ushort b) {
    uint u = uint(b) << 16;
    return as_type<float>(u);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// ── 4-bit dequant matrix-vector multiply ────────────────────────────────────
//
// Weight layout per row: [packed_u32 | scales_bf16 | biases_bf16]
// Each u32 packs 8 nibbles (4-bit values 0-15).
// Dequant: val = nibble * scale + bias  (affine per group)
//
// Grid: one threadgroup per output row (outDim threadgroups)
// Threads: 64 per threadgroup, reduce across inDim/8 u32 words
//
// Args:
//   0: weight buffer (u32, packed)
//   1: scale buffer  (bf16)
//   2: bias buffer   (bf16)
//   3: input buffer  (f32)
//   4: output buffer (f32)
//   5: outDim (uint)
//   6: inDim  (uint)
//   7: groupSize (uint)

kernel void matvec4bit(
    device const uint*   weights [[buffer(0)]],
    device const ushort* scales  [[buffer(1)]],
    device const ushort* biases  [[buffer(2)]],
    device const float*  input   [[buffer(3)]],
    device       float*  output  [[buffer(4)]],
    constant uint& outDim        [[buffer(5)]],
    constant uint& inDim         [[buffer(6)]],
    constant uint& groupSize     [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgSz [[threads_per_threadgroup]]
)
{
    if (row >= outDim) return;

    const uint numWords  = inDim / 8;
    const uint numGroups = inDim / groupSize;

    // Pointers for this row
    device const uint*   wRow = weights + row * numWords;
    device const ushort* sRow = scales  + row * numGroups;
    device const ushort* bRow = biases  + row * numGroups;

    float acc = 0.0f;

    for (uint w = lid; w < numWords; w += tgSz) {
        uint   packed = wRow[w];
        uint   g      = (w * 8) / groupSize;
        float  scale  = bf16_to_f32(sRow[g]);
        float  bias   = bf16_to_f32(bRow[g]);
        uint   base   = w * 8;

        // Unpack 8 nibbles and accumulate with FMA
        for (uint b = 0; b < 8; b++) {
            float nibble = float((packed >> (b * 4)) & 0xF);
            float x      = input[base + b];
            // fma(nibble, scale*x, bias*x)  — same FMA rearrangement from flash-moe
            acc = fma(nibble, scale * x, fma(bias, x, acc));
        }
    }

    // Threadgroup reduction
    threadgroup float shared[64];
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) output[row] = shared[0];
}

// ── RMS Norm (two-pass) ──────────────────────────────────────────────────────

// Pass 1: compute sum-of-squares into a single float
kernel void rmsNormSum(
    device const float*  x    [[buffer(0)]],
    device       float*  sumSq [[buffer(1)]],
    constant uint& dim        [[buffer(2)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgSz [[threads_per_threadgroup]]
)
{
    threadgroup float shared[256];
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tgSz) {
        float v = x[i];
        acc += v * v;
    }
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) sumSq[0] = shared[0];
}

// Pass 2: apply norm with BF16 weight
kernel void rmsNormApply(
    device const float*  x      [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    device const float*  sumSq  [[buffer(2)]],
    device       float*  out    [[buffer(3)]],
    constant uint&  dim         [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid >= dim) return;
    float rms = rsqrt(sumSq[0] / float(dim) + eps);
    out[gid] = x[gid] * rms * bf16_to_f32(weight[gid]);
}

// ── SwiGLU activation ────────────────────────────────────────────────────────

kernel void swiGLU(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device       float* act  [[buffer(2)]],
    constant uint& dim       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid >= dim) return;
    float g = gate[gid];
    act[gid] = (g / (1.0f + exp(-g))) * up[gid];   // SiLU(gate) * up
}

// ── Residual add ─────────────────────────────────────────────────────────────

kernel void residualAdd(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device       float* out [[buffer(2)]],
    constant uint& dim      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid >= dim) return;
    out[gid] = a[gid] + b[gid];
}

// ── RoPE (rotary position embeddings, in-place on Q and K) ──────────────────
//
// Grid: one thread per (head, pair) — i.e., numHeads × (rotaryDim/2) threads
// Q and K are interleaved in the same buffer: [Q | K]

kernel void ropeInPlace(
    device float*         qk        [[buffer(0)]],  // [qDim + kvDim] f32 in-place
    constant uint&        numQHeads [[buffer(1)]],
    constant uint&        numKHeads [[buffer(2)]],
    constant uint&        headDim   [[buffer(3)]],
    constant uint&        rotDim    [[buffer(4)]],
    constant float&       theta     [[buffer(5)]],
    constant uint&        pos       [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
)
{
    uint rotHalf = rotDim / 2;
    uint qDim    = numQHeads * headDim;
    uint totalElems = (numQHeads + numKHeads) * rotHalf;
    if (gid >= totalElems) return;

    // Which segment (Q vs K) and which head+pair
    bool isK   = (gid >= numQHeads * rotHalf);
    uint base  = isK ? qDim : 0;
    uint local = isK ? (gid - numQHeads * rotHalf) : gid;
    uint head  = local / rotHalf;
    uint pair  = local % rotHalf;
    uint hbase = base + head * headDim;

    float freq  = 1.0f / pow(theta, float(2 * pair) / float(rotDim));
    float angle = float(pos) * freq;
    float c = cos(angle), s = sin(angle);

    float x0 = qk[hbase + pair];
    float x1 = qk[hbase + pair + rotHalf];
    qk[hbase + pair]        = x0 * c - x1 * s;
    qk[hbase + pair + rotHalf] = x0 * s + x1 * c;
}

// ── Attention scores Q@K^T ───────────────────────────────────────────────────
//
// Grid: numHeads × seqLen threadgroups (one per (head, key_position))
// Each threadgroup computes one dot product: Q[head] · K[pos]

kernel void attnScores(
    device const float* q       [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],   // [seqLen × kvDim] row-major
    device       float* scores  [[buffer(2)]],   // [numHeads × seqLen]
    constant uint& headDim      [[buffer(3)]],
    constant uint& kvDim        [[buffer(4)]],
    constant uint& seqLen       [[buffer(5)]],
    constant uint& seqStride    [[buffer(6)]],   // pre-allocated stride
    constant float& scale       [[buffer(7)]],
    constant uint& hpkv         [[buffer(8)]],   // heads per kv-head
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgSz [[threads_per_threadgroup]]
)
{
    uint head = tgid / seqLen;
    uint pos  = tgid % seqLen;
    uint kvHead = head / hpkv;

    device const float* qh = q + head * headDim;
    device const float* kp = k_cache + pos * kvDim + kvHead * headDim;

    threadgroup float shared[64];
    float acc = 0.0f;
    for (uint i = lid; i < headDim; i += tgSz) acc += qh[i] * kp[i];
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) scores[head * seqStride + pos] = shared[0] * scale;
}

// ── Attention softmax (in-place, per head) ───────────────────────────────────

kernel void attnSoftmax(
    device float*      scores   [[buffer(0)]],
    constant uint&     seqLen   [[buffer(1)]],
    constant uint&     seqStride [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgSz [[threads_per_threadgroup]]
)
{
    device float* row = scores + head * seqStride;

    // Max reduction
    threadgroup float shared[256];
    float mx = -INFINITY;
    for (uint i = lid; i < seqLen; i += tgSz) mx = max(mx, row[i]);
    shared[lid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = shared[0];

    // Exp sum
    float sum = 0.0f;
    for (uint i = lid; i < seqLen; i += tgSz) {
        float e = exp(row[i] - maxVal);
        row[i] = e;
        sum += e;
    }
    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invSum = 1.0f / shared[0];

    // Normalize
    for (uint i = lid; i < seqLen; i += tgSz) row[i] *= invSum;
}

// ── Attention values: scores @ V ─────────────────────────────────────────────
//
// Grid: (numHeads × headDim + 255) / 256 threadgroups

kernel void attnValues(
    device const float* scores  [[buffer(0)]],
    device const float* v_cache [[buffer(1)]],   // [seqLen × kvDim]
    device       float* out     [[buffer(2)]],   // [numHeads × headDim]
    constant uint& headDim      [[buffer(3)]],
    constant uint& kvDim        [[buffer(4)]],
    constant uint& seqLen       [[buffer(5)]],
    constant uint& seqStride    [[buffer(6)]],
    constant uint& hpkv         [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
)
{
    uint head = gid / headDim;
    uint dim  = gid % headDim;
    uint kvHead = head / hpkv;

    device const float* sc = scores + head * seqStride;
    float acc = 0.0f;
    for (uint p = 0; p < seqLen; p++) {
        acc += sc[p] * v_cache[p * kvDim + kvHead * headDim + dim];
    }
    out[gid] = acc;
}

// ── BF16 matrix-vector multiply ──────────────────────────────────────────────
//
// Used for non-expert projections (attention QKV, o_proj, lm_head).
// Weight is stored BF16 in the mmap'd weight file; input/output are F32.
//
// Grid: one threadgroup per output row.  64 threads reduce across inDim.
// Args:
//   0: weight buffer (bf16, [outDim × inDim])
//   1: input  buffer (f32,  [inDim])
//   2: output buffer (f32,  [outDim])
//   3: outDim (uint)
//   4: inDim  (uint)

kernel void matvecBF16(
    device const ushort* weight [[buffer(0)]],
    device const float*  input  [[buffer(1)]],
    device       float*  output [[buffer(2)]],
    constant uint& outDim       [[buffer(3)]],
    constant uint& inDim        [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgSz [[threads_per_threadgroup]]
)
{
    if (row >= outDim) return;
    device const ushort* wRow = weight + row * inDim;
    float acc = 0.0f;
    for (uint i = lid; i < inDim; i += tgSz) {
        acc += bf16_to_f32(wRow[i]) * input[i];
    }
    threadgroup float shared[64];
    shared[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSz / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) output[row] = shared[0];
}

// ── MoE combine: weighted sum of expert outputs ──────────────────────────────
//
// Accumulates K expert outputs into moe_out using router weights.
// Grid: (hiddenDim + 63) / 64 threadgroups, 64 threads each

kernel void moeCombine(
    device const float*  expert_outs [[buffer(0)]],  // [K × hiddenDim]
    device const float*  weights     [[buffer(1)]],  // [K] routing weights
    device const float*  residual    [[buffer(2)]],  // [hiddenDim]
    device       float*  out         [[buffer(3)]],  // [hiddenDim]
    constant uint& K                 [[buffer(4)]],
    constant uint& hiddenDim         [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid >= hiddenDim) return;
    float acc = residual[gid];
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * hiddenDim + gid];
    }
    out[gid] = acc;
}
