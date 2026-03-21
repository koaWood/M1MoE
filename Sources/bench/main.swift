// bench — M1MoE inference benchmark
//
// Usage:
//   bench --config PATH [--model PATH] [--prompt TEXT]
//         [--tokens N] [--temp T] [--top-p P]

import Foundation
import M1MoE
import Darwin

// MARK: - Argument parsing

var _configPath: String? = nil
var _modelPath:  String? = nil
var _prompt      = "Hello, my name is"
var _numTokens   = 20
var _temp: Float = 0.0
var _topP: Float = 1.0

var argList = Array(CommandLine.arguments.dropFirst())
while !argList.isEmpty {
    let a = argList.removeFirst()
    switch a {
    case "--config": if let v = argList.first { _configPath = v; argList.removeFirst() }
    case "--model":  if let v = argList.first { _modelPath  = v; argList.removeFirst() }
    case "--prompt": if let v = argList.first { _prompt     = v; argList.removeFirst() }
    case "--tokens":
        if let v = argList.first, let n = Int(v) { _numTokens = n; argList.removeFirst() }
    case "--temp":
        if let v = argList.first, let f = Float(v) { _temp = f; argList.removeFirst() }
    case "--top-p":
        if let v = argList.first, let f = Float(v) { _topP = f; argList.removeFirst() }
    case "--help", "-h":
        print("""
            M1MoE bench — MoE inference benchmark

            --config PATH   Model JSON config (required)
            --model  PATH   Override model_dir from config
            --prompt TEXT   Input prompt (default: "Hello, my name is")
            --tokens N      Tokens to generate (default: 20)
            --temp   T      Sampling temperature (0 = greedy)
            --top-p  P      Nucleus threshold (default: 1.0)
            """)
        exit(0)
    default: break
    }
}

// MARK: - Timing

func nowMs() -> Double {
    var tv = timeval(); Darwin.gettimeofday(&tv, nil)
    return Double(tv.tv_sec) * 1000 + Double(tv.tv_usec) / 1000
}

// MARK: - Run on dedicated inference queue (blocks on GPU + SSD)

let inferQueue = DispatchQueue(label: "ai.m1moe.infer", qos: .userInteractive)
let done = DispatchSemaphore(value: 0)

inferQueue.async {
    guard let cfgPath = _configPath else {
        fputs("ERROR: --config PATH required\n", stderr); exit(1)
    }

    // ── Load config ──────────────────────────────────────────────────────────

    var spec: ModelSpec
    do { spec = try ModelSpec.load(from: URL(fileURLWithPath: cfgPath)) }
    catch { fputs("ERROR: config load failed: \(error)\n", stderr); exit(1) }
    if let ov = _modelPath { spec.modelDir = ov }

    guard let dir = spec.modelDir, !dir.isEmpty else {
        fputs("ERROR: no model_dir in config and --model not set\n", stderr); exit(1)
    }

    let modelURL = URL(fileURLWithPath: dir)
    let hw = HardwareProfile.detect()

    print("[config] \(spec.name) (\(spec.architecture))")
    print("[config] \(spec.numLayers) layers, \(spec.moe.numExperts) experts, top-\(spec.moe.topK)")
    print("[config] expert: \(spec.expertBytes4Bit / 1_048_576) MB each")
    print("[hw] \(hw.deviceName), \(hw.unifiedMemoryGB) GB RAM")
    print("[hw] expert window: \(hw.expertWindowCount(expertBytes: spec.expertBytes4Bit, numLayers: spec.numLayers, numExperts: spec.moe.numExperts)) experts")

    // ── Initialise subsystems ────────────────────────────────────────────────

    let t0 = nowMs()

    let weights: WeightStore
    do {
        weights = try WeightStore(
            binURL:  modelURL.appendingPathComponent("model_weights.bin"),
            jsonURL: modelURL.appendingPathComponent("model_weights.json")
        )
    } catch {
        fputs("ERROR: weights not found in \(dir) — run extract_olmoe.py first\n\n\(error)\n", stderr)
        exit(1)
    }

    let metal: MetalContext
    do { metal = try MetalContext(spec: spec, hardware: hw) }
    catch { fputs("ERROR: Metal init failed: \(error)\n", stderr); exit(1) }

    let engine: InferenceEngine
    do { engine = try InferenceEngine(spec: spec, metal: metal, weights: weights) }
    catch { fputs("ERROR: InferenceEngine init failed: \(error)\n", stderr); exit(1) }

    let dtInit = nowMs() - t0
    print(String(format: "[bench] initialised in %.0f ms", dtInit))

    // ── Tokenizer ────────────────────────────────────────────────────────────

    let tokURL = modelURL.appendingPathComponent("tokenizer.json")
    var tokenizer: Tokenizer? = nil
    var promptTokens: [UInt32]

    if FileManager.default.fileExists(atPath: tokURL.path) {
        do {
            tokenizer = try Tokenizer(url: tokURL)
            promptTokens = tokenizer!.encode(_prompt, addBOS: true,
                                             bos: spec.specialTokens.bos)
            print("[bench] prompt: \(promptTokens.count) tokens")
        } catch {
            fputs("WARN: tokenizer load failed (\(error)) — using BOS only\n", stderr)
            promptTokens = [spec.specialTokens.bos]
        }
    } else {
        print("[bench] tokenizer.json not found — using BOS token only")
        promptTokens = [spec.specialTokens.bos]
    }

    // ── Prefill ──────────────────────────────────────────────────────────────

    print("[bench] prefilling \(promptTokens.count) tokens …")
    let tPrefill = nowMs()

    var lastToken: UInt32 = spec.specialTokens.bos
    for tok in promptTokens {
        lastToken = engine.forward(token: tok, temperature: 0)
    }

    let dtPrefill = nowMs() - tPrefill
    print(String(format: "[bench] prefill: %.0f ms (%.1f ms/tok)",
                 dtPrefill, dtPrefill / Double(max(1, promptTokens.count))))

    // ── Generation ───────────────────────────────────────────────────────────

    print("[bench] generating \(_numTokens) tokens …\n")
    print(_prompt, terminator: "")
    fflush(stdout)

    var genTimes  = [Double]()
    var currentToken = lastToken

    for _ in 0 ..< _numTokens {
        let tTok = nowMs()
        let next = engine.forward(token: currentToken,
                                  temperature: _temp,
                                  topP: _topP)
        genTimes.append(nowMs() - tTok)

        if let tok = tokenizer {
            print(tok.decode([next], skipSpecial: true), terminator: "")
        } else {
            print(" \(next)", terminator: "")
        }
        fflush(stdout)

        currentToken = next
        if spec.specialTokens.eos.contains(currentToken) { break }
    }
    print()

    // ── Summary ───────────────────────────────────────────────────────────────

    let n = genTimes.count
    if n > 0 {
        let total = genTimes.reduce(0, +)
        let mean  = total / Double(n)
        let min_t = genTimes.min()!
        let max_t = genTimes.max()!
        print(String(format: "\n[bench] %d tokens: %.1f ms/tok (%.2f tok/s)  [min %.0f  max %.0f ms]",
                     n, mean, 1000.0 / mean, min_t, max_t))
    }

    done.signal()
}

done.wait()
