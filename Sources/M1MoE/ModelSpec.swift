// ModelSpec.swift — Complete model configuration loaded from JSON.
//
// All architecture parameters live here. No constants are hardcoded anywhere
// else in the engine — every kernel, buffer, and loop bound is derived from
// a ModelSpec instance at runtime.

import Foundation

// MARK: - ModelSpec

public struct ModelSpec: Codable, Sendable {

    // ── Identity ────────────────────────────────────────────────────────────
    public let name:         String
    public let architecture: String   // "olmoe" | "mixtral"
    public var modelDir:     String?  // overrideable via CLI

    // ── Core dims ───────────────────────────────────────────────────────────
    public let hiddenDim:    Int
    public let vocabSize:    Int
    public let numLayers:    Int
    public let rmsNormEps:   Float

    // ── Sub-configs ─────────────────────────────────────────────────────────
    public let attention:     AttentionSpec
    public let moe:           MoESpec
    public let quantization:  QuantSpec
    public let specialTokens: SpecialTokensSpec
    public let runtime:       RuntimeSpec
    public let chat:          ChatSpec?     // optional; defaults derived from architecture

    // ── Derived ─────────────────────────────────────────────────────────────

    /// Bytes per expert at 4-bit group quantization.
    /// Layout per expert: gate_proj [W|S|B] + up_proj [W|S|B] + down_proj [W|S|B]
    public var expertBytes4Bit: Int {
        let gs   = quantization.groupSize
        let h    = hiddenDim
        let i    = moe.intermediateDim

        // gate / up: [i, h] → weight [i, h/8] u32 + scales [i, h/gs] bf16 + biases [i, h/gs] bf16
        let gwSize  = i * (h / 8) * 4
        let gsbSize = i * (h / gs) * 2    // scales or biases

        // down: [h, i] → weight [h, i/8] u32 + scales [h, i/gs] bf16 + biases [h, i/gs] bf16
        let dwSize  = h * (i / 8) * 4
        let dsbSize = h * (i / gs) * 2

        return gwSize + gsbSize * 2 +   // gate
               gwSize + gsbSize * 2 +   // up
               dwSize + dsbSize * 2     // down
    }

    // ── Load ────────────────────────────────────────────────────────────────

    public static func load(from url: URL) throws -> ModelSpec {
        let data    = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ModelSpec.self, from: data)
    }
}

// MARK: - AttentionSpec

public struct AttentionSpec: Codable, Sendable {
    public let numHeads:          Int
    public let numKvHeads:        Int
    public let headDim:           Int
    public let ropeTheta:         Float
    public let maxSeqLen:         Int
    public let gpuKvSeqPrealloc:  Int

    public var qDim:   Int { numHeads   * headDim }
    public var kvDim:  Int { numKvHeads * headDim }
    public var hpkv:   Int { numHeads   / numKvHeads }  // heads per kv-head
    public var rotaryDim: Int { headDim }               // full rotary for these models
}

// MARK: - MoESpec

public struct MoESpec: Codable, Sendable {
    public let numExperts:      Int
    public let topK:            Int
    public let intermediateDim: Int
    public let normTopkProb:    Bool  // normalize routing weights to sum=1
}

// MARK: - QuantSpec

public struct QuantSpec: Codable, Sendable {
    public let bits:      Int
    public let groupSize: Int
}

// MARK: - SpecialTokensSpec

public struct SpecialTokensSpec: Codable, Sendable {
    public let eos: [UInt32]
    public let bos: UInt32
}

// MARK: - ChatSpec

public struct ChatSpec: Codable, Sendable {
    /// Text prepended to each user message
    public let userPrefix:      String
    /// Text appended to each user message (before assistant starts)
    public let userSuffix:      String
    /// Text prepended to each assistant turn (model continues from here)
    public let assistantPrefix: String
    /// Text appended to each completed assistant turn
    public let assistantSuffix: String
    /// Optional system prompt wrapper (nil = no system turn)
    public let systemPrefix:    String?
    public let systemSuffix:    String?
}

extension ModelSpec {
    /// Chat template — uses JSON-specified values if present, otherwise
    /// falls back to well-known defaults for the architecture.
    public var chatTemplate: ChatSpec {
        if let c = chat { return c }
        switch architecture {
        case "mixtral":
            // Mixtral-Instruct v0.1 format
            return ChatSpec(
                userPrefix:      "[INST] ",
                userSuffix:      " [/INST]",
                assistantPrefix: " ",
                assistantSuffix: "</s>",
                systemPrefix:    nil,
                systemSuffix:    nil
            )
        default:
            // OLMoE / generic instruct format
            return ChatSpec(
                userPrefix:      "<|user|>\n",
                userSuffix:      "\n<|assistant|>\n",
                assistantPrefix: "",
                assistantSuffix: "<|endoftext|>\n",
                systemPrefix:    "<|system|>\n",
                systemSuffix:    "\n"
            )
        }
    }
}

// MARK: - RuntimeSpec

public struct RuntimeSpec: Codable, Sendable {
    /// Number of recent tokens whose expert activations to keep in DRAM.
    /// Implements the Apple paper's "windowing" strategy.
    public let expertWindowSize: Int

    /// Parallel I/O threads for expert pread.
    public let ioThreads: Int
}
