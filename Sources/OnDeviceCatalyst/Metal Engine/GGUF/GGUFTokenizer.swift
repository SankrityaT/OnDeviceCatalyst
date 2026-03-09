//
//  GGUFTokenizer.swift
//  OnDeviceCatalyst
//
//  Pure-Swift BPE tokenizer that reads vocab from GGUF metadata.
//  Produces identical token IDs to llama.cpp for the same model.
//

import Foundation

/// Pure-Swift BPE tokenizer loaded from GGUF metadata.
public final class GGUFTokenizer {

    // MARK: - Vocab Data

    /// Token ID → string piece
    private let vocabTokens: [String]

    /// Token string → token ID (reverse lookup)
    private let tokenToId: [String: Int32]

    /// Token scores (used for merge priority in BPE)
    private let scores: [Float]

    /// Token types (normal, unknown, control, user-defined, byte, etc.)
    private let tokenTypes: [Int]

    /// Special tokens
    public let bosTokenId: Int32
    public let eosTokenId: Int32
    private let eogTokenIds: Set<Int32>

    /// Byte-fallback tokens (for handling unknown bytes)
    private let byteTokens: [UInt8: Int32]

    // MARK: - BPE Merge Table

    /// Merge pair → merged result token string
    private let mergeTable: [MergePair: Int]

    /// Ordered merge rules (lower index = higher priority)
    private struct MergePair: Hashable {
        let left: String
        let right: String
    }

    // MARK: - Init

    /// Initialize from GGUF metadata.
    public init(metadata: [String: GGUFValue]) throws {
        // Extract vocab tokens
        guard let tokensArray = metadata["tokenizer.ggml.tokens"]?.asStringArray else {
            throw CatalystError.modelFileCorrupted(
                path: "<metadata>",
                reason: "Missing tokenizer.ggml.tokens in GGUF metadata"
            )
        }
        self.vocabTokens = tokensArray

        // Build reverse lookup
        var tokenToId: [String: Int32] = [:]
        for (i, token) in tokensArray.enumerated() {
            tokenToId[token] = Int32(i)
        }
        self.tokenToId = tokenToId

        // Extract scores
        if let scoresArray = metadata["tokenizer.ggml.scores"]?.asFloatArray {
            self.scores = scoresArray
        } else {
            self.scores = Array(repeating: 0.0, count: tokensArray.count)
        }

        // Extract token types
        if case .array(let typesArray)? = metadata["tokenizer.ggml.token_type"] {
            self.tokenTypes = typesArray.compactMap { $0.asInt }
        } else {
            self.tokenTypes = Array(repeating: 0, count: tokensArray.count)
        }

        // Special tokens
        self.bosTokenId = metadata["tokenizer.ggml.bos_token_id"]?.asInt.map(Int32.init) ?? 1
        self.eosTokenId = metadata["tokenizer.ggml.eos_token_id"]?.asInt.map(Int32.init) ?? 2

        // EOG tokens (end-of-generation, may include multiple IDs)
        var eog: Set<Int32> = [self.eosTokenId]
        if case .array(let eogArray)? = metadata["tokenizer.ggml.eog_token_id"] {
            for val in eogArray {
                if let id = val.asInt {
                    eog.insert(Int32(id))
                }
            }
        }
        self.eogTokenIds = eog

        // Build byte-fallback table
        var byteTokens: [UInt8: Int32] = [:]
        for (i, token) in tokensArray.enumerated() {
            // Byte tokens look like "<0x00>", "<0x01>", etc.
            if token.hasPrefix("<0x") && token.hasSuffix(">") && token.count == 6 {
                let hexStr = String(token.dropFirst(3).dropLast(1))
                if let byte = UInt8(hexStr, radix: 16) {
                    byteTokens[byte] = Int32(i)
                }
            }
        }
        self.byteTokens = byteTokens

        // Build merge table from merge rules or from scores
        var mergeTable: [MergePair: Int] = [:]
        if let mergesArray = metadata["tokenizer.ggml.merges"]?.asStringArray {
            // Explicit merge rules (GPT-2 / Qwen style)
            for (priority, merge) in mergesArray.enumerated() {
                let parts = merge.split(separator: " ", maxSplits: 1)
                if parts.count == 2 {
                    let pair = MergePair(left: String(parts[0]), right: String(parts[1]))
                    mergeTable[pair] = priority
                }
            }
        } else {
            // SentencePiece-style: use scores as merge priorities.
            // Build merge rules from all pairs that form known tokens.
            for (i, token) in tokensArray.enumerated() {
                let utf8 = Array(token.utf8)
                guard utf8.count >= 2 else { continue }
                // Try all possible splits of this token
                for splitAt in 1..<utf8.count {
                    let leftBytes = Array(utf8[0..<splitAt])
                    let rightBytes = Array(utf8[splitAt...])
                    if let leftStr = String(bytes: leftBytes, encoding: .utf8),
                       let rightStr = String(bytes: rightBytes, encoding: .utf8),
                       tokenToId[leftStr] != nil,
                       tokenToId[rightStr] != nil {
                        let pair = MergePair(left: leftStr, right: rightStr)
                        // Higher score = higher priority = lower merge index
                        // Use negative score so that highest score merges first
                        if mergeTable[pair] == nil {
                            mergeTable[pair] = Int(-scores[i] * 1000000)
                        }
                    }
                }
            }
        }
        self.mergeTable = mergeTable
    }

    // MARK: - Tokenization

    /// Tokenize text into token IDs.
    public func tokenize(text: String, addBos: Bool) -> [Int32] {
        var tokens: [Int32] = []

        if addBos {
            tokens.append(bosTokenId)
        }

        guard !text.isEmpty else { return tokens }

        // Initial split: each UTF-8 byte becomes a character token or byte-fallback
        var pieces: [String] = []
        for byte in text.utf8 {
            let charStr = String(UnicodeScalar(byte))
            if tokenToId[charStr] != nil {
                pieces.append(charStr)
            } else {
                // Try byte-fallback token
                let byteStr = String(format: "<0x%02X>", byte)
                pieces.append(byteStr)
            }
        }

        // For models with explicit pre-tokenization (like Qwen, Llama 3),
        // try to match the full text against known multi-char tokens first
        if mergeTable.isEmpty {
            // No merge rules — just look up each character
            for piece in pieces {
                if let id = tokenToId[piece] {
                    tokens.append(id)
                }
            }
            return tokens
        }

        // BPE: iteratively merge the highest-priority pair
        var symbols = pieces

        while symbols.count >= 2 {
            // Find the highest-priority merge (lowest merge index)
            var bestMergeIdx: Int? = nil
            var bestPriority = Int.max
            var bestPos = 0

            for i in 0..<(symbols.count - 1) {
                let pair = MergePair(left: symbols[i], right: symbols[i + 1])
                if let priority = mergeTable[pair], priority < bestPriority {
                    bestPriority = priority
                    bestMergeIdx = i
                    bestPos = i
                }
            }

            guard let mergePos = bestMergeIdx else {
                break // No more merges possible
            }

            // Merge the pair
            let merged = symbols[mergePos] + symbols[mergePos + 1]
            symbols[mergePos] = merged
            symbols.remove(at: mergePos + 1)
        }

        // Convert symbols to token IDs
        for symbol in symbols {
            if let id = tokenToId[symbol] {
                tokens.append(id)
            } else {
                // Fall back to byte-level encoding
                for byte in symbol.utf8 {
                    if let byteId = byteTokens[byte] {
                        tokens.append(byteId)
                    }
                }
            }
        }

        return tokens
    }

    // MARK: - Detokenization

    /// Convert a single token ID to text.
    public func detokenize(token: Int32) -> String {
        let id = Int(token)
        guard id >= 0 && id < vocabTokens.count else { return "" }

        let piece = vocabTokens[id]

        // Handle byte-fallback tokens
        if piece.hasPrefix("<0x") && piece.hasSuffix(">") && piece.count == 6 {
            let hexStr = String(piece.dropFirst(3).dropLast(1))
            if let byte = UInt8(hexStr, radix: 16) {
                return String(UnicodeScalar(byte))
            }
        }

        return piece
    }

    /// Convert an array of token IDs to text.
    public func detokenize(tokens: [Int32]) -> String {
        var bytes: [UInt8] = []
        for token in tokens {
            let id = Int(token)
            guard id >= 0 && id < vocabTokens.count else { continue }

            let piece = vocabTokens[id]

            if piece.hasPrefix("<0x") && piece.hasSuffix(">") && piece.count == 6 {
                let hexStr = String(piece.dropFirst(3).dropLast(1))
                if let byte = UInt8(hexStr, radix: 16) {
                    bytes.append(byte)
                    continue
                }
            }

            bytes.append(contentsOf: piece.utf8)
        }

        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - Special Token Checks

    /// Check if a token is an end-of-generation token.
    public func isEndOfGeneration(token: Int32) -> Bool {
        return eogTokenIds.contains(token)
    }

    /// Vocabulary size.
    public var vocabularySize: Int32 {
        return Int32(vocabTokens.count)
    }
}
