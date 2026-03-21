// chat — M1MoE interactive multi-turn chat REPL
//
// Usage:
//   chat --config PATH [--model PATH] [--system TEXT]
//        [--temp T] [--top-p P] [--max-ctx N]
//
// Commands during chat:
//   /reset    — clear conversation history and KV cache
//   /system S — set a new system prompt (clears history)
//   /stats    — print last token timing
//   /quit     — exit

import Foundation
import M1MoE
import Darwin

// MARK: - Arguments

var _configPath: String? = nil
var _modelPath:  String? = nil
var _systemMsg:  String? = nil
var _temp: Float = 0.7
var _topP: Float = 0.9
var _maxCtx:     Int = 2048   // slide window when context exceeds this

var argList = Array(CommandLine.arguments.dropFirst())
while !argList.isEmpty {
    let a = argList.removeFirst()
    switch a {
    case "--config": if let v = argList.first { _configPath = v; argList.removeFirst() }
    case "--model":  if let v = argList.first { _modelPath  = v; argList.removeFirst() }
    case "--system": if let v = argList.first { _systemMsg  = v; argList.removeFirst() }
    case "--temp":   if let v = argList.first, let f = Float(v)  { _temp = f; argList.removeFirst() }
    case "--top-p":  if let v = argList.first, let f = Float(v)  { _topP = f; argList.removeFirst() }
    case "--max-ctx":if let v = argList.first, let n = Int(v)    { _maxCtx = n; argList.removeFirst() }
    case "--help", "-h":
        print("""
            M1MoE chat — interactive multi-turn conversation

            --config  PATH   Model JSON config (required)
            --model   PATH   Override model_dir from config
            --system  TEXT   System prompt text
            --temp    T      Sampling temperature (default: 0.7)
            --top-p   P      Nucleus threshold (default: 0.9)
            --max-ctx N      Max context tokens before sliding window (default: 2048)

            During chat:
              /reset         Clear history and KV cache
              /system TEXT   Set new system prompt
              /stats         Show last-turn timing
              /quit          Exit
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

// MARK: - Main (dedicated inference queue)

let inferQueue = DispatchQueue(label: "ai.m1moe.chat", qos: .userInteractive)
let done = DispatchSemaphore(value: 0)

inferQueue.async {

    guard let cfgPath = _configPath else {
        fputs("ERROR: --config PATH required\n", stderr); exit(1)
    }

    // ── Load config ──────────────────────────────────────────────────────────

    var spec: ModelSpec
    do { spec = try ModelSpec.load(from: URL(fileURLWithPath: cfgPath)) }
    catch { fputs("ERROR: \(error)\n", stderr); exit(1) }
    if let ov = _modelPath { spec.modelDir = ov }

    guard let dir = spec.modelDir, !dir.isEmpty else {
        fputs("ERROR: no model_dir — set in config or pass --model\n", stderr); exit(1)
    }
    let modelURL = URL(fileURLWithPath: dir)
    let hw = HardwareProfile.detect()

    // ── Init ─────────────────────────────────────────────────────────────────

    let t0 = nowMs()

    let weights: WeightStore
    do {
        weights = try WeightStore(
            binURL:  modelURL.appendingPathComponent("model_weights.bin"),
            jsonURL: modelURL.appendingPathComponent("model_weights.json")
        )
    } catch { fputs("ERROR: weights: \(error)\n", stderr); exit(1) }

    let metal: MetalContext
    do { metal = try MetalContext(spec: spec, hardware: hw) }
    catch { fputs("ERROR: Metal: \(error)\n", stderr); exit(1) }

    let engine: InferenceEngine
    do { engine = try InferenceEngine(spec: spec, metal: metal, weights: weights) }
    catch { fputs("ERROR: engine: \(error)\n", stderr); exit(1) }

    let tokURL = modelURL.appendingPathComponent("tokenizer.json")
    guard FileManager.default.fileExists(atPath: tokURL.path) else {
        fputs("ERROR: tokenizer.json not found in \(dir)\n", stderr); exit(1)
    }
    let tokenizer: Tokenizer
    do { tokenizer = try Tokenizer(url: tokURL) }
    catch { fputs("ERROR: tokenizer: \(error)\n", stderr); exit(1) }

    print(String(format: "[chat] %@ ready in %.0f ms  (temp=%.1f top-p=%.1f)",
                 spec.name, nowMs() - t0, _temp, _topP))

    // ── Load optional system prompt from file ─────────────────────────────────

    let sysFile = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent(".m1moe/system.md")
    var systemPrompt: String? = _systemMsg
    if systemPrompt == nil, FileManager.default.fileExists(atPath: sysFile.path) {
        systemPrompt = try? String(contentsOf: sysFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let s = systemPrompt {
            print("[chat] system prompt loaded from ~/.m1moe/system.md (\(s.count) chars)")
        }
    }

    // ── Conversation state ────────────────────────────────────────────────────

    let tmpl = spec.chatTemplate

    // All tokens accumulated across turns (used for KV cache continuations)
    var contextTokens = [UInt32]()
    var lastStats: (tokCount: Int, ms: Double) = (0, 0)

    /// Append tokens to context, prefill them, return token count prefilled.
    @discardableResult
    func prefillTokens(_ tokens: [UInt32]) -> Int {
        for tok in tokens {
            engine.forward(token: tok, temperature: 0)
        }
        contextTokens.append(contentsOf: tokens)
        return tokens.count
    }

    /// Full reset — clear KV cache and context.
    func resetConversation() {
        engine.reset()
        contextTokens.removeAll()
        print("[chat] conversation reset")
    }

    /// Re-prefill entire context from scratch (used after sliding window truncation).
    func reprefillContext() {
        engine.reset()
        for tok in contextTokens {
            engine.forward(token: tok, temperature: 0)
        }
    }

    // Prefill system prompt if present
    if let sys = systemPrompt,
       let pre = tmpl.systemPrefix, let suf = tmpl.systemSuffix {
        let text = pre + sys + suf
        prefillTokens(tokenizer.encode(text, addBOS: false))
    }

    // ── REPL ─────────────────────────────────────────────────────────────────

    print("Type your message. Commands: /reset /system /stats /quit\n")

    while true {

        // ── Read user input ──────────────────────────────────────────────────

        print("You: ", terminator: "")
        fflush(stdout)

        guard let line = readLine(strippingNewline: true) else { break }   // EOF
        let input = line.trimmingCharacters(in: .whitespaces)
        guard !input.isEmpty else { continue }

        // ── Handle slash commands ────────────────────────────────────────────

        if input.hasPrefix("/") {
            let parts = input.dropFirst().split(separator: " ", maxSplits: 1)
            let cmd = parts.first.map(String.init) ?? ""
            let arg = parts.count > 1 ? String(parts[1]) : ""

            switch cmd {
            case "quit", "exit", "q":
                print("Bye.")
                done.signal(); return

            case "reset":
                resetConversation()
                if let sys = systemPrompt,
                   let pre = tmpl.systemPrefix, let suf = tmpl.systemSuffix {
                    prefillTokens(tokenizer.encode(pre + sys + suf, addBOS: false))
                }
                continue

            case "system":
                guard !arg.isEmpty else { print("[chat] usage: /system <prompt>"); continue }
                systemPrompt = arg
                resetConversation()
                if let pre = tmpl.systemPrefix, let suf = tmpl.systemSuffix {
                    prefillTokens(tokenizer.encode(pre + arg + suf, addBOS: false))
                }
                print("[chat] system prompt updated")
                continue

            case "stats":
                if lastStats.tokCount > 0 {
                    let ms = lastStats.ms / Double(lastStats.tokCount)
                    print(String(format: "[chat] last turn: %d tokens, %.1f ms/tok (%.2f tok/s)",
                                 lastStats.tokCount, ms, 1000.0 / ms))
                } else {
                    print("[chat] no stats yet")
                }
                continue

            default:
                print("[chat] unknown command: /\(cmd)")
                continue
            }
        }

        // ── Sliding window: truncate if approaching max context ───────────────

        let userText = tmpl.userPrefix + input + tmpl.userSuffix
        let userTokens = tokenizer.encode(userText, addBOS: false)

        if contextTokens.count + userTokens.count + 256 > _maxCtx {
            // Drop oldest turns (keep ~half the window), preserve from first token
            let keep = _maxCtx / 2
            if contextTokens.count > keep {
                contextTokens = Array(contextTokens.suffix(keep))
                print(String(format: "[chat] context trimmed to %d tokens", contextTokens.count))
                reprefillContext()
            }
        }

        // ── Prefill user turn ────────────────────────────────────────────────

        prefillTokens(userTokens)

        // ── Generate assistant response ───────────────────────────────────────

        print("Bot: \(tmpl.assistantPrefix)", terminator: "")
        fflush(stdout)

        // Prefill assistantPrefix if non-empty
        if !tmpl.assistantPrefix.isEmpty {
            prefillTokens(tokenizer.encode(tmpl.assistantPrefix, addBOS: false))
        }

        var genTokens  = [UInt32]()
        var genMs      = [Double]()
        var responseText = ""

        // Stop sequences — prevent model from continuing into next user turn
        let stopStrings: [String] = [
            tmpl.userPrefix.trimmingCharacters(in: .whitespacesAndNewlines),
            "<|user|>", "[INST]", "<|endoftext|>"
        ].filter { !$0.isEmpty }

        generationLoop: while true {
            let tTok = nowMs()
            let tok = engine.forward(token: genTokens.last ?? contextTokens.last ?? spec.specialTokens.bos,
                                     temperature: _temp, topP: _topP)
            genMs.append(nowMs() - tTok)
            genTokens.append(tok)

            // EOS check
            if spec.specialTokens.eos.contains(tok) { break }

            // Decode and stream
            let piece = tokenizer.decode([tok], skipSpecial: true)
            responseText += piece
            print(piece, terminator: "")
            fflush(stdout)

            // Stop sequence check (rolling last 32 chars of response)
            let tail = String(responseText.suffix(64))
            for stop in stopStrings {
                if tail.contains(stop) {
                    // Trim the stop string from printed output
                    // (already printed, so just break — minor cosmetic issue)
                    break generationLoop
                }
            }

            // Safety: respect max context
            if contextTokens.count + genTokens.count >= _maxCtx { break }
        }

        print()  // newline after response

        // Accumulate generated tokens into context
        contextTokens.append(contentsOf: genTokens)

        // Append assistant suffix to context (don't prefill it — it's a delimiter)
        if !tmpl.assistantSuffix.isEmpty {
            let suffixToks = tokenizer.encode(tmpl.assistantSuffix, addBOS: false)
            prefillTokens(suffixToks)
        }

        // Stats
        lastStats = (genTokens.count, genMs.reduce(0, +))
        if genMs.count > 1 {
            let mean = genMs.reduce(0, +) / Double(genMs.count)
            print(String(format: "  [%.0f ms/tok · %.1f tok/s · ctx %d]",
                         mean, 1000.0 / mean, contextTokens.count))
        }
    }

    done.signal()
}

done.wait()
