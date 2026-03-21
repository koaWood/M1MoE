// MetalContext.swift — GPU device, pipeline states, and scratch buffers.
//
// All buffer sizes are derived from ModelSpec + HardwareProfile.
// No constants hardcoded here.

import Metal
import Foundation

// MARK: - MetalContext

public final class MetalContext: @unchecked Sendable {

    public let device:   MTLDevice
    public let queue:    MTLCommandQueue
    public let hardware: HardwareProfile

    // ── Pipeline state objects ──────────────────────────────────────────────
    // matvec
    public let pso_matvec4:      MTLComputePipelineState  // 4-bit dequant matmul
    public let pso_matvecBF16:   MTLComputePipelineState  // BF16 matmul (attention projections)
    // norm
    public let pso_rms_sum:      MTLComputePipelineState
    public let pso_rms_apply:    MTLComputePipelineState
    // activation
    public let pso_swiglu:       MTLComputePipelineState
    // attention
    public let pso_attn_scores:  MTLComputePipelineState
    public let pso_attn_softmax: MTLComputePipelineState
    public let pso_attn_values:  MTLComputePipelineState
    // rope
    public let pso_rope:         MTLComputePipelineState
    // residual
    public let pso_residual_add: MTLComputePipelineState
    // moe combine
    public let pso_moe_combine:  MTLComputePipelineState

    // ── Scratch buffers (derived from ModelSpec) ────────────────────────────
    public let buf_input:      MTLBuffer   // [hiddenDim] f32 — current token embedding
    public let buf_residual:   MTLBuffer   // [hiddenDim] f32
    public let buf_h_mid:      MTLBuffer   // [hiddenDim] f32 — post-attn residual
    public let buf_output:     MTLBuffer   // [hiddenDim] f32 — o_proj result
    public let buf_norm_sq:    MTLBuffer   // [1] f32 — RMS norm accumulator
    public let buf_logits:     MTLBuffer   // [vocabSize] f32

    // Attention
    public let buf_q:          MTLBuffer   // [numHeads × headDim] f32
    public let buf_k:          MTLBuffer   // [numKvHeads × headDim] f32
    public let buf_v:          MTLBuffer   // [numKvHeads × headDim] f32
    public let buf_attn_out:   MTLBuffer   // [numHeads × headDim] f32
    public let buf_attn_scores: MTLBuffer  // [numHeads × gpuKvSeqPrealloc] f32
    public let kvK:            [MTLBuffer] // per-layer K cache [gpuKvSeq × kvDim]
    public let kvV:            [MTLBuffer] // per-layer V cache [gpuKvSeq × kvDim]

    // Expert compute (top-K slots)
    public let expertBufs:      [MTLBuffer]  // [K] — loaded expert data (2MB-aligned)
    public let expertGate:      [MTLBuffer]  // [K] — gate_proj outputs [intermediateDim]
    public let expertUp:        [MTLBuffer]  // [K] — up_proj outputs
    public let expertAct:       [MTLBuffer]  // [K] — SwiGLU outputs
    public let expertOutPacked: MTLBuffer    // [topK × hiddenDim] — down_proj outputs (packed)
    public let buf_expert_in:   MTLBuffer    // [hiddenDim] — shared expert input
    public let buf_router_w:    MTLBuffer    // [topK] f32  — routing weights for moeCombine

    // MoE routing
    public let buf_gate_scores: MTLBuffer  // [numExperts] f32
    public let buf_moe_out:     MTLBuffer  // [hiddenDim] f32 — combined MoE output

    // MARK: - Init

    public init(spec: ModelSpec, hardware: HardwareProfile) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDevice
        }
        guard let q = dev.makeCommandQueue() else {
            throw MetalError.noQueue
        }
        self.device   = dev
        self.queue    = q
        self.hardware = hardware

        // Compile shaders from bundled Shaders.metal source at runtime.
        // SPM copies the .metal file into the bundle as-is; we compile it here
        // using MTLDevice.makeLibrary(source:options:).
        let t0 = Date()
        guard let shaderURL = Bundle.module.url(forResource: "Shaders", withExtension: "metal"),
              let src = try? String(contentsOf: shaderURL, encoding: .utf8) else {
            throw MetalError.missingFunction("Shaders.metal not found in bundle")
        }
        let opts = MTLCompileOptions()
        opts.fastMathEnabled = true
        let lib = try dev.makeLibrary(source: src, options: opts)
        let dt  = Date().timeIntervalSince(t0) * 1000
        print(String(format: "[metal] %@ — shaders compiled in %.0f ms", dev.name, dt))

        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            return try dev.makeComputePipelineState(function: fn)
        }

        pso_matvec4      = try pso("matvec4bit")
        pso_matvecBF16   = try pso("matvecBF16")
        pso_rms_sum      = try pso("rmsNormSum")
        pso_rms_apply    = try pso("rmsNormApply")
        pso_swiglu       = try pso("swiGLU")
        pso_attn_scores  = try pso("attnScores")
        pso_attn_softmax = try pso("attnSoftmax")
        pso_attn_values  = try pso("attnValues")
        pso_rope         = try pso("ropeInPlace")
        pso_residual_add = try pso("residualAdd")
        pso_moe_combine  = try pso("moeCombine")

        // ── Scratch buffers ────────────────────────────────────────────────

        let F  = MemoryLayout<Float>.size   // 4
        let h  = spec.hiddenDim
        let v  = spec.vocabSize
        let K  = spec.moe.topK
        let I  = spec.moe.intermediateDim
        let E  = spec.moe.numExperts
        let nh = spec.attention.numHeads
        let nk = spec.attention.numKvHeads
        let hd = spec.attention.headDim
        let sl = spec.attention.gpuKvSeqPrealloc
        let nl = spec.numLayers

        func buf(_ bytes: Int, _ label: String) -> MTLBuffer {
            let b = dev.makeBuffer(length: max(bytes, 4),
                                   options: .storageModeShared)!
            b.label = label
            return b
        }

        buf_input       = buf(h * F,          "input")
        buf_residual    = buf(h * F,          "residual")
        buf_h_mid       = buf(h * F,          "h_mid")
        buf_output      = buf(h * F,          "output")
        buf_norm_sq     = buf(F,              "norm_sq")
        buf_logits      = buf(v * F,          "logits")

        buf_q           = buf(nh * hd * F,    "q")
        buf_k           = buf(nk * hd * F,    "k")
        buf_v           = buf(nk * hd * F,    "v")
        buf_attn_out    = buf(nh * hd * F,    "attn_out")
        buf_attn_scores = buf(nh * sl * F,    "attn_scores")

        kvK = (0 ..< nl).map { buf(sl * nk * hd * F, "kvK[\($0)]") }
        kvV = (0 ..< nl).map { buf(sl * nk * hd * F, "kvV[\($0)]") }

        buf_expert_in   = buf(h * F,          "expert_in")
        buf_gate_scores = buf(E * F,          "gate_scores")
        buf_moe_out     = buf(h * F,          "moe_out")

        // Expert slots: 2MB-aligned backing for DMA-friendly pread
        let eBytes  = spec.expertBytes4Bit
        let eAlloc  = (eBytes + (2 << 20) - 1) & ~((2 << 20) - 1)
        var eBufs   = [MTLBuffer]()
        var gBufs   = [MTLBuffer]()
        var uBufs   = [MTLBuffer]()
        var aBufs   = [MTLBuffer]()
        for k in 0 ..< K {
            var raw: UnsafeMutableRawPointer? = nil
            Darwin.posix_memalign(&raw, 2 << 20, eAlloc)
            Darwin.memset(raw!, 0, eAlloc)
            let eb = dev.makeBuffer(bytesNoCopy: raw!, length: eAlloc,
                                    options: .storageModeShared, deallocator: nil)!
            eb.label = "expert_data[\(k)]"
            eBufs.append(eb)
            gBufs.append(buf(I * F, "expert_gate[\(k)]"))
            uBufs.append(buf(I * F, "expert_up[\(k)]"))
            aBufs.append(buf(I * F, "expert_act[\(k)]"))
        }
        expertBufs      = eBufs
        expertGate      = gBufs
        expertUp        = uBufs
        expertAct       = aBufs
        expertOutPacked = buf(K * h * F, "expert_out_packed")
        buf_router_w    = buf(K * F,     "router_weights")

        let kvMB = Double(sl * nk * hd * F * nl * 2) / 1_048_576
        print(String(format: "[metal] KV cache: %d layers × %.1f MB = %.0f MB",
                     nl, Double(sl * nk * hd * F * 2) / 1_048_576, kvMB))
        print(String(format: "[metal] Expert slots: %d × %.1f MB",
                     K, Double(eAlloc) / 1_048_576))
    }
}

// MARK: - Errors

public enum MetalError: Error {
    case noDevice
    case noQueue
    case missingFunction(String)
}
