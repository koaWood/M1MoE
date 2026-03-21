// InferenceEngine.swift — Token-by-token MoE forward pass.
//
// Supports OLMoE ("olmoe") and Mixtral ("mixtral") architectures.
// All dimensions are read from ModelSpec — no hardcoded constants.
//
// Expert loading: direct pread() into Metal-shared buffers; OS page cache
// handles hot-expert retention ("Trust the OS" principle from flash-moe).
//
// Execution is fully synchronous — call forward() from a background thread
// or DispatchQueue; do NOT call from the main thread.

import Darwin
import Foundation
import Metal

// MARK: - InferenceEngine

public final class InferenceEngine {

    public let spec:    ModelSpec
    public let metal:   MetalContext
    public let weights: WeightStore

    // Sequence position (0-based, incremented per token)
    public private(set) var seqPos: Int = 0

    // Architecture flags
    private let hasQKNorm: Bool

    // Expert file descriptors [numLayers] — opened at init, closed at deinit
    private let expertFDs: [Int32]

    // Expert layout: byte offsets within one expert block
    private let gateWOff:  Int
    private let gateSCOff: Int
    private let gateBCOff: Int
    private let upWOff:    Int
    private let upSCOff:   Int
    private let upBCOff:   Int
    private let downWOff:  Int
    private let downSCOff: Int
    private let downBCOff: Int

    // Pre-converted router weights [numLayers][numExperts × hiddenDim] f32
    private let routerF32: [[Float]]

    // MARK: - Init

    public init(spec: ModelSpec,
                metal: MetalContext,
                weights: WeightStore) throws {
        self.spec    = spec
        self.metal   = metal
        self.weights = weights
        self.hasQKNorm = (spec.architecture == "olmoe")

        // Zero-copy Metal buffer from mmap'd weight file
        guard weights.makeMetalBuffer(device: metal.device) != nil else {
            throw EngineError.weightBufferFailed
        }

        // Expert file descriptors
        let dir = URL(fileURLWithPath: spec.modelDir ?? ".")
            .appendingPathComponent("packed_experts")
        var fds = [Int32](repeating: -1, count: spec.numLayers)
        for L in 0 ..< spec.numLayers {
            let path = dir.appendingPathComponent(
                String(format: "layer_%02d.bin", L)).path
            let fd = Darwin.open(path, O_RDONLY)
            if fd >= 0 {
                _ = Darwin.fcntl(fd, F_RDAHEAD, 0)   // no read-ahead; OS caches on demand
                fds[L] = fd
            }
        }
        let open = fds.filter { $0 >= 0 }.count
        print("[engine] \(open)/\(spec.numLayers) expert layer files open")
        self.expertFDs = fds

        // Expert layout byte offsets (must match extract_olmoe.py packing)
        let h  = spec.hiddenDim
        let i  = spec.moe.intermediateDim
        let gs = spec.quantization.groupSize

        let gw  = i * (h / 8) * 4
        let gsb = i * (h / gs) * 2
        let dw  = h * (i / 8) * 4
        let dsb = h * (i / gs) * 2

        gateWOff  = 0
        gateSCOff = gateWOff  + gw
        gateBCOff = gateSCOff + gsb
        upWOff    = gateBCOff + gsb
        upSCOff   = upWOff    + gw
        upBCOff   = upSCOff   + gsb
        downWOff  = upBCOff   + gsb
        downSCOff = downWOff  + dw
        downBCOff = downSCOff + dsb

        // Pre-load router weights as F32
        let numExperts = spec.moe.numExperts
        var allRouters = [[Float]]()
        for L in 0 ..< spec.numLayers {
            let key = WeightKeys.router(architecture: spec.architecture, layer: L)
            guard let ptr = weights.bf16(named: key) else {
                throw EngineError.missingWeight(key)
            }
            var row = [Float](repeating: 0, count: numExperts * h)
            for idx in 0 ..< numExperts * h {
                row[idx] = bf16ToF32(ptr[idx])
            }
            allRouters.append(row)
        }
        self.routerF32 = allRouters
    }

    deinit {
        for fd in expertFDs where fd >= 0 { Darwin.close(fd) }
    }

    // MARK: - Forward Pass

    /// Compute one forward step. Returns the predicted next token.
    /// MUST be called from a background thread (blocks on GPU + disk I/O).
    @discardableResult
    public func forward(token: UInt32, temperature: Float = 0, topP: Float = 1) -> UInt32 {
        let h  = spec.hiddenDim
        let hF = h * MemoryLayout<Float>.size

        // 1. Embed token → buf_residual
        embedToken(token)
        memcpy(metal.buf_residual.contents(), metal.buf_input.contents(), hF)

        // 2. Transformer layers — capture position once, pass to every layer
        let pos = seqPos
        for L in 0 ..< spec.numLayers {
            runLayer(L, pos: pos)
        }
        seqPos += 1

        // 3. Final RMS norm + lm_head
        let cmd = makeCmd()
        let enc = cmd.makeComputeCommandEncoder()!
        encodeRMSNorm(enc, input: metal.buf_residual,
                      weight: "model.norm.weight", output: metal.buf_input)
        encodeMV16(enc, weight: "lm_head.weight",
                   input: metal.buf_input, output: metal.buf_logits,
                   outDim: spec.vocabSize, inDim: h)
        enc.endEncoding()
        commitSync(cmd)

        // 4. Sample
        let logitsPtr = metal.buf_logits.contents()
            .bindMemory(to: Float.self, capacity: spec.vocabSize)
        return sampleToken(logits: logitsPtr, count: spec.vocabSize,
                           temperature: temperature, topP: topP)
    }

    /// Reset KV cache and sequence position for a new conversation.
    public func reset() {
        seqPos = 0
        for buf in metal.kvK { memset(buf.contents(), 0, buf.length) }
        for buf in metal.kvV { memset(buf.contents(), 0, buf.length) }
    }

    // MARK: - Layer

    private func runLayer(_ L: Int, pos: Int) {
        let h   = spec.hiddenDim
        let hF  = h * MemoryLayout<Float>.size
        let qd  = spec.attention.qDim
        let kvd = spec.attention.kvDim
        let nh  = spec.attention.numHeads
        let nk  = spec.attention.numKvHeads
        let hd  = spec.attention.headDim
        let sl  = pos + 1

        // ── CMD1: input norm + QKV projections ──────────────────────────────
        let cmd1 = makeCmd()
        let enc1 = cmd1.makeComputeCommandEncoder()!

        encodeRMSNorm(enc1,
                      input:  metal.buf_residual,
                      weight: WeightKeys.inputNorm(architecture: spec.architecture, layer: L),
                      output: metal.buf_input)
        encodeMV16(enc1, weight: WeightKeys.qProj(layer: L),
                   input: metal.buf_input, output: metal.buf_q,  outDim: qd,  inDim: h)
        encodeMV16(enc1, weight: WeightKeys.kProj(layer: L),
                   input: metal.buf_input, output: metal.buf_k,  outDim: kvd, inDim: h)
        encodeMV16(enc1, weight: WeightKeys.vProj(layer: L),
                   input: metal.buf_input, output: metal.buf_v,  outDim: kvd, inDim: h)
        enc1.endEncoding()
        commitSync(cmd1)

        // ── CPU: QK norms + RoPE + KV cache store ───────────────────────────
        let qPtr = metal.buf_q.contents().bindMemory(to: Float.self, capacity: qd)
        let kPtr = metal.buf_k.contents().bindMemory(to: Float.self, capacity: kvd)
        let vPtr = metal.buf_v.contents().bindMemory(to: Float.self, capacity: kvd)

        if hasQKNorm {
            if let qw = weights.bf16(named: WeightKeys.qNorm(layer: L)) {
                applyHeadRMSNorm(buffer: qPtr, numHeads: nh, headDim: hd,
                                 weight: qw, eps: spec.rmsNormEps)
            }
            if let kw = weights.bf16(named: WeightKeys.kNorm(layer: L)) {
                applyHeadRMSNorm(buffer: kPtr, numHeads: nk, headDim: hd,
                                 weight: kw, eps: spec.rmsNormEps)
            }
        }

        applyRoPE(buffer: qPtr, numHeads: nh, headDim: hd,
                  theta: spec.attention.ropeTheta, pos: pos)
        applyRoPE(buffer: kPtr, numHeads: nk, headDim: hd,
                  theta: spec.attention.ropeTheta, pos: pos)

        let kvkBase = metal.kvK[L].contents()
            .advanced(by: pos * kvd * MemoryLayout<Float>.size)
        let kvvBase = metal.kvV[L].contents()
            .advanced(by: pos * kvd * MemoryLayout<Float>.size)
        memcpy(kvkBase, kPtr, kvd * MemoryLayout<Float>.size)
        memcpy(kvvBase, vPtr, kvd * MemoryLayout<Float>.size)

        // ── CMD2: attention + o_proj + residual + post-norm ─────────────────
        let cmd2 = makeCmd()
        let enc2 = cmd2.makeComputeCommandEncoder()!

        encodeAttention(enc2, layer: L, seqLen: sl)
        encodeMV16(enc2, weight: WeightKeys.oProj(layer: L),
                   input: metal.buf_attn_out, output: metal.buf_output,
                   outDim: h, inDim: qd)
        encodeResidualAdd(enc2,
                          a: metal.buf_residual, b: metal.buf_output,
                          out: metal.buf_residual, dim: h)
        encodeRMSNorm(enc2,
                      input:  metal.buf_residual,
                      weight: WeightKeys.postAttnNorm(architecture: spec.architecture, layer: L),
                      output: metal.buf_h_mid)
        enc2.endEncoding()
        commitSync(cmd2)

        // ── CPU: router + expert selection ──────────────────────────────────
        let normedPtr = metal.buf_h_mid.contents()
            .bindMemory(to: Float.self, capacity: h)
        let scores = computeRouterScores(layer: L, input: normedPtr)
        let topK   = selectTopK(scores: scores, k: spec.moe.topK)

        // Routing weights → GPU buffer
        let rwPtr = metal.buf_router_w.contents()
            .bindMemory(to: Float.self, capacity: spec.moe.topK)
        for (slot, (_, w)) in topK.enumerated() { rwPtr[slot] = w }

        // ── Expert load + compute ─────────────────────────────────────────
        for (slot, (expertIdx, _)) in topK.enumerated() {
            loadExpert(layer: L, index: expertIdx, slot: slot)
        }

        let cmd3 = makeCmd()
        let enc3 = cmd3.makeComputeCommandEncoder()!
        for (slot, _) in topK.enumerated() {
            encodeExpertForward(enc3, slot: slot)
        }
        encodeMoECombine(enc3, topK: spec.moe.topK, dim: h)
        enc3.endEncoding()
        commitSync(cmd3)

        // moe_out → residual
        memcpy(metal.buf_residual.contents(), metal.buf_moe_out.contents(), hF)
    }

    // MARK: - Expert Loading (direct pread, no actor overhead)

    private func loadExpert(layer: Int, index: Int, slot: Int) {
        let dst  = metal.expertBufs[slot].contents()
        let size = spec.expertBytes4Bit
        guard layer < expertFDs.count, expertFDs[layer] >= 0 else {
            memset(dst, 0, size); return
        }
        let offset = off_t(index) * off_t(size)
        let n = Darwin.pread(expertFDs[layer], dst, size, offset)
        if n != size { memset(dst, 0, size) }
    }

    // MARK: - Metal Command Helpers

    private func makeCmd() -> MTLCommandBuffer {
        metal.queue.makeCommandBuffer()!
    }

    /// Commit and block until complete — safe to call from a background thread.
    private func commitSync(_ cmd: MTLCommandBuffer) {
        let sema = DispatchSemaphore(value: 0)
        cmd.addCompletedHandler { _ in sema.signal() }
        cmd.commit()
        sema.wait()
    }

    // MARK: - Metal Encoders

    private func encodeRMSNorm(_ enc: MTLComputeCommandEncoder,
                                input: MTLBuffer, weight: String, output: MTLBuffer) {
        let h = spec.hiddenDim
        enc.setComputePipelineState(metal.pso_rms_sum)
        enc.setBuffer(input,             offset: 0, index: 0)
        enc.setBuffer(metal.buf_norm_sq, offset: 0, index: 1)
        var dim = UInt32(h)
        enc.setBytes(&dim, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1,height: 1,depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256,height: 1,depth: 1))

        enc.setComputePipelineState(metal.pso_rms_apply)
        enc.setBuffer(input,             offset: 0, index: 0)
        let wOff = weights.offset(named: weight) ?? 0
        enc.setBuffer(weights.wfBuf!,    offset: wOff, index: 1)
        enc.setBuffer(metal.buf_norm_sq, offset: 0, index: 2)
        enc.setBuffer(output,            offset: 0, index: 3)
        enc.setBytes(&dim, length: 4, index: 4)
        var eps = spec.rmsNormEps
        enc.setBytes(&eps, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: h,height: 1,depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
    }

    private func encodeMV16(_ enc: MTLComputeCommandEncoder,
                             weight: String, input: MTLBuffer, output: MTLBuffer,
                             outDim: Int, inDim: Int) {
        enc.setComputePipelineState(metal.pso_matvecBF16)
        let wOff = weights.offset(named: weight) ?? 0
        enc.setBuffer(weights.wfBuf!, offset: wOff, index: 0)
        enc.setBuffer(input,          offset: 0,    index: 1)
        enc.setBuffer(output,         offset: 0,    index: 2)
        var od = UInt32(outDim), id = UInt32(inDim)
        enc.setBytes(&od, length: 4, index: 3)
        enc.setBytes(&id, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: outDim,height: 1,depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
    }

    private func encodeResidualAdd(_ enc: MTLComputeCommandEncoder,
                                    a: MTLBuffer, b: MTLBuffer, out: MTLBuffer, dim: Int) {
        enc.setComputePipelineState(metal.pso_residual_add)
        enc.setBuffer(a,   offset: 0, index: 0)
        enc.setBuffer(b,   offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var d = UInt32(dim)
        enc.setBytes(&d, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: dim,height: 1,depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
    }

    private func encodeAttention(_ enc: MTLComputeCommandEncoder, layer: Int, seqLen: Int) {
        let nh   = spec.attention.numHeads
        let hd   = spec.attention.headDim
        let kvd  = spec.attention.kvDim
        let sl   = spec.attention.gpuKvSeqPrealloc
        let hpkv = spec.attention.hpkv
        let scale = 1.0 / sqrtf(Float(hd))

        var hd32   = UInt32(hd),  kvd32 = UInt32(kvd)
        var sl32   = UInt32(seqLen), sls32 = UInt32(sl)
        var sc32   = scale
        var hpkv32 = UInt32(hpkv)

        // attnScores
        enc.setComputePipelineState(metal.pso_attn_scores)
        enc.setBuffer(metal.buf_q,           offset: 0, index: 0)
        enc.setBuffer(metal.kvK[layer],      offset: 0, index: 1)
        enc.setBuffer(metal.buf_attn_scores, offset: 0, index: 2)
        enc.setBytes(&hd32,   length: 4, index: 3)
        enc.setBytes(&kvd32,  length: 4, index: 4)
        enc.setBytes(&sl32,   length: 4, index: 5)
        enc.setBytes(&sls32,  length: 4, index: 6)
        enc.setBytes(&sc32,   length: 4, index: 7)
        enc.setBytes(&hpkv32, length: 4, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: nh * seqLen, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        // attnSoftmax
        enc.setComputePipelineState(metal.pso_attn_softmax)
        enc.setBuffer(metal.buf_attn_scores, offset: 0, index: 0)
        enc.setBytes(&sl32,  length: 4, index: 1)
        enc.setBytes(&sls32, length: 4, index: 2)
        enc.dispatchThreadgroups(
            MTLSize(width: nh, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // attnValues
        enc.setComputePipelineState(metal.pso_attn_values)
        enc.setBuffer(metal.buf_attn_scores, offset: 0, index: 0)
        enc.setBuffer(metal.kvV[layer],      offset: 0, index: 1)
        enc.setBuffer(metal.buf_attn_out,    offset: 0, index: 2)
        enc.setBytes(&hd32,   length: 4, index: 3)
        enc.setBytes(&kvd32,  length: 4, index: 4)
        enc.setBytes(&sl32,   length: 4, index: 5)
        enc.setBytes(&sls32,  length: 4, index: 6)
        enc.setBytes(&hpkv32, length: 4, index: 7)
        enc.dispatchThreads(
            MTLSize(width: nh * hd, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    private func encodeExpertForward(_ enc: MTLComputeCommandEncoder, slot: Int) {
        let h  = spec.hiddenDim
        let i  = spec.moe.intermediateDim
        let gs = UInt32(spec.quantization.groupSize)
        let eb = metal.expertBufs[slot]

        // gate_proj
        encode4BitMV(enc,
                     wBuf: eb, wOff: gateWOff,
                     sBuf: eb, sOff: gateSCOff,
                     bBuf: eb, bOff: gateBCOff,
                     input: metal.buf_h_mid,
                     output: metal.expertGate[slot],
                     outDim: UInt32(i), inDim: UInt32(h), gs: gs)

        // up_proj
        encode4BitMV(enc,
                     wBuf: eb, wOff: upWOff,
                     sBuf: eb, sOff: upSCOff,
                     bBuf: eb, bOff: upBCOff,
                     input: metal.buf_h_mid,
                     output: metal.expertUp[slot],
                     outDim: UInt32(i), inDim: UInt32(h), gs: gs)

        // SwiGLU
        enc.setComputePipelineState(metal.pso_swiglu)
        enc.setBuffer(metal.expertGate[slot], offset: 0, index: 0)
        enc.setBuffer(metal.expertUp[slot],   offset: 0, index: 1)
        enc.setBuffer(metal.expertAct[slot],  offset: 0, index: 2)
        var d = UInt32(i)
        enc.setBytes(&d, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: i,height: 1,depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))

        // down_proj → expertOutPacked at offset slot * h * 4
        encode4BitMV(enc,
                     wBuf: eb, wOff: downWOff,
                     sBuf: eb, sOff: downSCOff,
                     bBuf: eb, bOff: downBCOff,
                     input: metal.expertAct[slot],
                     output: metal.expertOutPacked,
                     outputOffset: slot * h * MemoryLayout<Float>.size,
                     outDim: UInt32(h), inDim: UInt32(i), gs: gs)
    }

    private func encode4BitMV(_ enc: MTLComputeCommandEncoder,
                               wBuf: MTLBuffer, wOff: Int,
                               sBuf: MTLBuffer, sOff: Int,
                               bBuf: MTLBuffer, bOff: Int,
                               input: MTLBuffer,
                               output: MTLBuffer, outputOffset: Int = 0,
                               outDim: UInt32, inDim: UInt32, gs: UInt32) {
        enc.setComputePipelineState(metal.pso_matvec4)
        enc.setBuffer(wBuf,   offset: wOff,        index: 0)
        enc.setBuffer(sBuf,   offset: sOff,        index: 1)
        enc.setBuffer(bBuf,   offset: bOff,        index: 2)
        enc.setBuffer(input,  offset: 0,            index: 3)
        enc.setBuffer(output, offset: outputOffset, index: 4)
        var od = outDim, id = inDim, g = gs
        enc.setBytes(&od, length: 4, index: 5)
        enc.setBytes(&id, length: 4, index: 6)
        enc.setBytes(&g,  length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: Int(outDim),height: 1,depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
    }

    private func encodeMoECombine(_ enc: MTLComputeCommandEncoder, topK: Int, dim: Int) {
        enc.setComputePipelineState(metal.pso_moe_combine)
        enc.setBuffer(metal.expertOutPacked, offset: 0, index: 0)
        enc.setBuffer(metal.buf_router_w,    offset: 0, index: 1)
        enc.setBuffer(metal.buf_residual,    offset: 0, index: 2)
        enc.setBuffer(metal.buf_moe_out,     offset: 0, index: 3)
        var k32 = UInt32(topK), d32 = UInt32(dim)
        enc.setBytes(&k32, length: 4, index: 4)
        enc.setBytes(&d32, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: dim,height: 1,depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64,height: 1,depth: 1))
    }

    // MARK: - CPU Operations

    private func embedToken(_ token: UInt32) {
        let h = spec.hiddenDim
        guard let ptr = weights.bf16(named: "model.embed_tokens.weight") else { return }
        let row = ptr.advanced(by: Int(token) * h)
        let out = metal.buf_input.contents().bindMemory(to: Float.self, capacity: h)
        for d in 0 ..< h { out[d] = bf16ToF32(row[d]) }
    }

    private func computeRouterScores(layer: Int, input: UnsafePointer<Float>) -> [Float] {
        let h = spec.hiddenDim
        let n = spec.moe.numExperts
        let w = routerF32[layer]
        var scores = [Float](repeating: 0, count: n)
        for e in 0 ..< n {
            var dot: Float = 0
            let base = e * h
            for d in 0 ..< h { dot += w[base + d] * input[d] }
            scores[e] = dot
        }
        // Softmax
        let mx = scores.max()!
        var sum: Float = 0
        for e in 0 ..< n { scores[e] = expf(scores[e] - mx); sum += scores[e] }
        let inv = 1.0 / sum
        for e in 0 ..< n { scores[e] *= inv }
        return scores
    }

    private func selectTopK(scores: [Float], k: Int) -> [(Int, Float)] {
        var indexed = scores.enumerated().map { ($0.offset, $0.element) }
        indexed.sort { $0.1 > $1.1 }
        var result = Array(indexed.prefix(k))
        if spec.moe.normTopkProb {
            let s = result.reduce(0) { $0 + $1.1 }
            result = result.map { ($0.0, $0.1 / s) }
        }
        return result
    }

    private func applyRoPE(buffer: UnsafeMutablePointer<Float>,
                            numHeads: Int, headDim: Int, theta: Float, pos: Int) {
        let half = headDim / 2
        for head in 0 ..< numHeads {
            let base = head * headDim
            for pair in 0 ..< half {
                let freq  = 1.0 / powf(theta, Float(2 * pair) / Float(headDim))
                let angle = Float(pos) * freq
                let c = cosf(angle), s = sinf(angle)
                let x0 = buffer[base + pair]
                let x1 = buffer[base + pair + half]
                buffer[base + pair]        = x0 * c - x1 * s
                buffer[base + pair + half] = x0 * s + x1 * c
            }
        }
    }

    private func applyHeadRMSNorm(buffer: UnsafeMutablePointer<Float>,
                                   numHeads: Int, headDim: Int,
                                   weight: UnsafePointer<UInt16>, eps: Float) {
        for head in 0 ..< numHeads {
            let base = head * headDim
            var ss: Float = 0
            for d in 0 ..< headDim { ss += buffer[base + d] * buffer[base + d] }
            let rms = 1.0 / sqrtf(ss / Float(headDim) + eps)
            for d in 0 ..< headDim { buffer[base + d] *= rms * bf16ToF32(weight[d]) }
        }
    }

    private func sampleToken(logits: UnsafePointer<Float>,
                              count: Int, temperature: Float, topP: Float) -> UInt32 {
        if temperature <= 0 {
            var best = 0; var bestV = logits[0]
            for i in 1 ..< count { if logits[i] > bestV { bestV = logits[i]; best = i } }
            return UInt32(best)
        }
        var probs = [Float](repeating: 0, count: count)
        var mx = logits[0]
        for i in 1 ..< count { if logits[i] > mx { mx = logits[i] } }
        var sum: Float = 0
        for i in 0 ..< count { probs[i] = expf((logits[i] - mx) / temperature); sum += probs[i] }
        let inv = 1.0 / sum
        for i in 0 ..< count { probs[i] *= inv }

        let sorted = probs.enumerated().sorted { $0.element > $1.element }
        var cumul: Float = 0; var nucleus: [(Int, Float)] = []
        for (idx, p) in sorted {
            nucleus.append((idx, p)); cumul += p
            if cumul >= topP { break }
        }
        let ns = nucleus.reduce(0) { $0 + $1.1 }
        var r = Float.random(in: 0 ..< ns)
        for (idx, p) in nucleus { r -= p; if r <= 0 { return UInt32(idx) } }
        return UInt32(nucleus.last!.0)
    }
}

// MARK: - Weight key naming

private enum WeightKeys {
    static func inputNorm(architecture: String, layer: Int) -> String {
        "model.layers.\(layer).input_layernorm.weight"
    }
    static func postAttnNorm(architecture: String, layer: Int) -> String {
        "model.layers.\(layer).post_attention_layernorm.weight"
    }
    static func qProj(layer: Int) -> String { "model.layers.\(layer).self_attn.q_proj.weight" }
    static func kProj(layer: Int) -> String { "model.layers.\(layer).self_attn.k_proj.weight" }
    static func vProj(layer: Int) -> String { "model.layers.\(layer).self_attn.v_proj.weight" }
    static func oProj(layer: Int) -> String { "model.layers.\(layer).self_attn.o_proj.weight" }
    static func qNorm(layer: Int) -> String { "model.layers.\(layer).self_attn.q_norm.weight" }
    static func kNorm(layer: Int) -> String { "model.layers.\(layer).self_attn.k_norm.weight" }
    static func router(architecture: String, layer: Int) -> String {
        switch architecture {
        case "mixtral": return "model.layers.\(layer).block_sparse_moe.gate.weight"
        default:        return "model.layers.\(layer).mlp.gate.weight"
        }
    }
}

// MARK: - BF16 helper

@inline(__always)
private func bf16ToF32(_ v: UInt16) -> Float {
    Float(bitPattern: UInt32(v) << 16)
}

// MARK: - Errors

public enum EngineError: Error {
    case weightBufferFailed
    case missingWeight(String)
}
