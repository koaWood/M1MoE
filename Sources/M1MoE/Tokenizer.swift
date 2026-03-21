// Tokenizer.swift — BPE tokenizer supporting two modes:
//
//   byteLevelBPE   — GPT-2 / GPT-NeoX / OLMoE
//                    Bytes encoded as unicode chars (Ġ = space, etc.)
//                    Detected when model.byte_fallback is absent or false.
//
//   sentencePieceBPE — Mistral / Mixtral / LLaMA
//                    Spaces encoded as ▁ (U+2581), unknown bytes as <0xNN>.
//                    Detected when model.byte_fallback == true.
//
// Both modes read the same HuggingFace tokenizer.json format.

import Foundation

// MARK: - Tokenizer

public final class Tokenizer {

    private enum Mode { case byteLevelBPE, sentencePieceBPE }

    private let vocab:      [String: UInt32]
    private let idToToken:  [UInt32: String]
    private let mergeRank:  [MergePair: Int]
    private let mode:       Mode

    // GPT-2 byte ↔ unicode tables (byteLevelBPE only)
    private static let byteEncoder: [UInt8: String]  = makeByteLevelEncoder()
    private static let byteDecoder: [String: UInt8]  = {
        Dictionary(uniqueKeysWithValues: makeByteLevelEncoder().map { ($1, $0) })
    }()

    // MARK: - Init

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root     = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model    = root["model"] as? [String: Any],
              let vocabRaw = model["vocab"] as? [String: Any],
              let mergesRaw = model["merges"] as? [String]
        else { throw TokenizerError.invalidFormat }

        // Detect mode
        let byteFallback = model["byte_fallback"] as? Bool ?? false
        self.mode = byteFallback ? .sentencePieceBPE : .byteLevelBPE

        // Vocab
        var v = [String: UInt32]()
        for (k, val) in vocabRaw {
            if let id = val as? Int { v[k] = UInt32(id) }
        }
        self.vocab     = v
        self.idToToken = Dictionary(uniqueKeysWithValues: v.map { ($1, $0) })

        // Merge ranks
        var ranks = [MergePair: Int]()
        for (rank, merge) in mergesRaw.enumerated() {
            let parts = merge.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            ranks[MergePair(String(parts[0]), String(parts[1]))] = rank
        }
        self.mergeRank = ranks
    }

    // MARK: - Encode

    public func encode(_ text: String, addBOS: Bool = true, bos: UInt32 = 1) -> [UInt32] {
        guard !text.isEmpty else { return addBOS ? [bos] : [] }
        switch mode {
        case .byteLevelBPE:     return encodeByteLevelBPE(text, addBOS: addBOS, bos: bos)
        case .sentencePieceBPE: return encodeSentencePiece(text, addBOS: addBOS, bos: bos)
        }
    }

    // MARK: - Decode

    public func decode(_ ids: [UInt32], skipSpecial: Bool = true) -> String {
        switch mode {
        case .byteLevelBPE:     return decodeByteLevelBPE(ids, skipSpecial: skipSpecial)
        case .sentencePieceBPE: return decodeSentencePiece(ids, skipSpecial: skipSpecial)
        }
    }

    // MARK: - Byte-level BPE (GPT-2 / OLMoE)

    private func encodeByteLevelBPE(_ text: String, addBOS: Bool, bos: UInt32) -> [UInt32] {
        var ids = addBOS ? [bos] : [UInt32]()

        // Split on whitespace boundaries; prefix non-first words with Ġ (space marker)
        var words = [String]()
        var current = ""
        for (i, ch) in text.unicodeScalars.enumerated() {
            let isSpace = ch.value == 0x20
            if isSpace && !current.isEmpty {
                words.append(current)
                current = "Ġ"
            } else if i == 0 && isSpace {
                current = "Ġ"
            } else {
                let s = String(ch)
                for byte in s.utf8 {
                    current += Self.byteEncoder[byte] ?? "?"
                }
            }
        }
        if !current.isEmpty { words.append(current) }

        for word in words {
            for sym in applyBPE(word.map { String($0) }) {
                if let id = vocab[sym] { ids.append(id) }
            }
        }
        return ids
    }

    private func decodeByteLevelBPE(_ ids: [UInt32], skipSpecial: Bool) -> String {
        var bytes = [UInt8]()
        for id in ids {
            guard let tok = idToToken[id] else { continue }
            for ch in tok.unicodeScalars {
                let s = String(ch)
                if let b = Self.byteDecoder[s] { bytes.append(b) }
            }
        }
        return String(bytes: bytes, encoding: .utf8)
            ?? String(bytes: bytes, encoding: .isoLatin1)
            ?? ""
    }

    // MARK: - SentencePiece BPE (Mistral / Mixtral / LLaMA)

    private func encodeSentencePiece(_ text: String, addBOS: Bool, bos: UInt32) -> [UInt32] {
        var ids = addBOS ? [bos] : [UInt32]()

        // Metaspace pre-tokeniser: split on spaces, prepend ▁ to every piece
        let words = text.components(separatedBy: " ")
        for (i, word) in words.enumerated() {
            guard !word.isEmpty else { continue }
            let piece = (i == 0 ? "▁" : "▁") + word   // always add ▁ prefix

            // BPE on the piece
            let syms = applyBPE(splitSentencePieceChars(piece))
            for sym in syms {
                if let id = vocab[sym] {
                    ids.append(id)
                } else {
                    // Byte fallback: encode each UTF-8 byte as <0xNN>
                    for byte in sym.utf8 {
                        let tok = String(format: "<0x%02X>", byte)
                        if let id = vocab[tok] { ids.append(id) }
                    }
                }
            }
        }
        return ids
    }

    /// Split a SentencePiece word into initial BPE symbols, respecting
    /// multi-byte unicode characters as single symbols (unlike byte-level where
    /// each byte is a symbol).
    private func splitSentencePieceChars(_ word: String) -> [String] {
        // Each Unicode scalar is one initial symbol; ▁ is treated as a single symbol too
        var syms = [String]()
        for scalar in word.unicodeScalars {
            syms.append(String(scalar))
        }
        return syms
    }

    private func decodeSentencePiece(_ ids: [UInt32], skipSpecial: Bool) -> String {
        var bytes = [UInt8]()
        for id in ids {
            guard let tok = idToToken[id] else { continue }

            if isSentencePieceSpecial(tok) {
                if !skipSpecial { bytes.append(contentsOf: tok.utf8) }
                continue
            }

            // Byte fallback token: <0x0A>, <0x41>, etc.
            if tok.hasPrefix("<0x") && tok.hasSuffix(">") && tok.count == 6 {
                let hexStr = String(tok.dropFirst(3).dropLast())
                if let b = UInt8(hexStr, radix: 16) {
                    bytes.append(b)
                    continue
                }
            }

            // Regular token: ▁ → space, rest as UTF-8
            let s = tok.replacingOccurrences(of: "▁", with: " ")
            bytes.append(contentsOf: s.utf8)
        }
        return String(bytes: bytes, encoding: .utf8)
            ?? String(bytes: bytes, encoding: .isoLatin1)
            ?? ""
    }

    /// Returns true for special control tokens that should be skipped in normal output.
    private func isSentencePieceSpecial(_ tok: String) -> Bool {
        switch tok {
        case "<s>", "</s>", "<unk>", "<pad>": return true
        default:
            // <|...|> style special tokens
            return tok.hasPrefix("<|") && tok.hasSuffix("|>")
        }
    }

    // MARK: - BPE (shared)

    private func applyBPE(_ symbols: [String]) -> [String] {
        var syms = symbols
        while syms.count > 1 {
            var bestRank = Int.max
            var bestIdx  = -1
            for i in 0 ..< syms.count - 1 {
                if let rank = mergeRank[MergePair(syms[i], syms[i + 1])], rank < bestRank {
                    bestRank = rank
                    bestIdx  = i
                }
            }
            if bestIdx == -1 { break }
            syms[bestIdx] = syms[bestIdx] + syms[bestIdx + 1]
            syms.remove(at: bestIdx + 1)
        }
        return syms
    }

    // MARK: - GPT-2 byte encoder table

    private static func makeByteLevelEncoder() -> [UInt8: String] {
        var bs = [Int]()
        for c in 33...126 { bs.append(c) }
        for c in 161...172 { bs.append(c) }
        for c in 174...255 { bs.append(c) }
        var cs = bs
        var n = 0
        for b in 0...255 {
            if !bs.contains(b) { bs.append(b); cs.append(256 + n); n += 1 }
        }
        var table = [UInt8: String]()
        for (b, c) in zip(bs, cs) {
            if let scalar = Unicode.Scalar(c) { table[UInt8(b)] = String(scalar) }
        }
        return table
    }
}

// MARK: - Helpers

private struct MergePair: Hashable {
    let a: String; let b: String
    init(_ a: String, _ b: String) { self.a = a; self.b = b }
}

// MARK: - Errors

public enum TokenizerError: Error {
    case invalidFormat
    case fileNotFound(String)
}
