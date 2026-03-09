//
//  LlamaCppBackend.swift
//  OnDeviceCatalyst
//
//  InferenceBackend implementation wrapping the existing LlamaBridge / llama.cpp.
//

import Foundation
import llama

/// Wraps the existing `LlamaBridge` static methods into the `InferenceBackend` protocol.
public final class LlamaCppBackend: InferenceBackend {

    // MARK: - Internal State

    private var model: CModel?
    private var context: CContext?
    private var batch: CBatch?
    private var batchCapacity: Int = 512
    private var backendInitialized = false

    // MARK: - Lifecycle

    public func loadModel(path: String, settings: InstanceSettings) throws {
        if !backendInitialized {
            LlamaBridge.initializeBackend()
            backendInitialized = true
        }
        model = try LlamaBridge.loadModel(path: path, settings: settings)
    }

    public func createContext(settings: InstanceSettings) throws {
        guard let model = model else {
            throw CatalystError.engineNotInitialized
        }
        context = try LlamaBridge.createContext(model: model, settings: settings)
        batch = try LlamaBridge.createBatch(maxTokens: settings.batchSize)
        batchCapacity = Int(settings.batchSize)
    }

    public func warmup() throws {
        guard let model = model, let context = context, var batch = batch else {
            throw CatalystError.engineNotInitialized
        }

        let tokens = try LlamaBridge.tokenize(text: "Hello", model: model, addBos: true)
        guard !tokens.isEmpty else {
            throw CatalystError.tokenizationFailed(text: "Hello", reason: "No tokens generated")
        }

        LlamaBridge.clearBatch(&batch)
        LlamaBridge.addTokenToBatch(
            batch: &batch,
            token: tokens[0],
            position: 0,
            sequenceId: 0,
            generateLogits: true
        )
        try LlamaBridge.processBatch(context: context, batch: batch)
        LlamaBridge.clearKVCache(context)
    }

    public func shutdown() {
        if let batch = batch {
            LlamaBridge.freeBatch(batch)
            self.batch = nil
        }
        if let context = context {
            LlamaBridge.freeContext(context)
            self.context = nil
        }
        if let model = model {
            LlamaBridge.freeModel(model)
            self.model = nil
        }
    }

    deinit {
        shutdown()
    }

    // MARK: - Model Metadata

    public var vocabularySize: Int32 {
        guard let model = model else { return 0 }
        return LlamaBridge.getVocabularySize(model)
    }

    public var embeddingSize: Int32 {
        guard let model = model else { return 0 }
        return LlamaBridge.getEmbeddingSize(model)
    }

    public func isEndOfGeneration(token: Int32) -> Bool {
        guard let model = model else { return false }
        return LlamaBridge.isEndOfGeneration(model, token: token)
    }

    // MARK: - Tokenization

    public func tokenize(text: String, addBos: Bool, parseSpecial: Bool = true) throws -> [Int32] {
        guard let model = model else {
            throw CatalystError.engineNotInitialized
        }
        return try LlamaBridge.tokenize(text: text, model: model, addBos: addBos, parseSpecial: parseSpecial)
    }

    public func detokenize(token: Int32) -> String {
        guard let model = model else { return "" }
        return LlamaBridge.detokenize(token: token, model: model)
    }

    // MARK: - Prompt Processing

    public func processTokens(_ tokens: [Int32], positions: [Int32], generateLogitsAtLast: Bool) throws {
        guard let context = context, var batch = batch else {
            throw CatalystError.engineNotInitialized
        }

        var tokenIndex = 0
        while tokenIndex < tokens.count {
            LlamaBridge.clearBatch(&batch)

            let end = min(tokenIndex + batchCapacity, tokens.count)
            for i in tokenIndex..<end {
                let isLast = (i == tokens.count - 1)
                let generateLogits = isLast && generateLogitsAtLast
                LlamaBridge.addTokenToBatch(
                    batch: &batch,
                    token: tokens[i],
                    position: positions[i],
                    sequenceId: 0,
                    generateLogits: generateLogits
                )
            }

            try LlamaBridge.processBatch(context: context, batch: batch)
            tokenIndex = end
        }

        self.batch = batch
    }

    // MARK: - Decode Step

    public func decodeToken(_ token: Int32, position: Int32) throws {
        guard let context = context, var batch = batch else {
            throw CatalystError.engineNotInitialized
        }

        LlamaBridge.clearBatch(&batch)
        LlamaBridge.addTokenToBatch(
            batch: &batch,
            token: token,
            position: position,
            sequenceId: 0,
            generateLogits: true
        )
        try LlamaBridge.processBatch(context: context, batch: batch)
        self.batch = batch
    }

    // MARK: - Logits

    public func getLogits() -> UnsafeMutablePointer<Float>? {
        guard let context = context else { return nil }
        return LlamaBridge.getLogits(context: context, batchIndex: -1)
    }

    // MARK: - KV Cache

    public func clearKVCache() {
        guard let context = context else { return }
        LlamaBridge.clearKVCache(context)
    }

    public func removeKVCacheTokens(sequenceId: Int32, startPos: Int32, endPos: Int32) {
        guard let context = context else { return }
        LlamaBridge.removeKVCacheTokens(
            context: context,
            sequenceId: sequenceId,
            startPos: startPos,
            endPos: endPos
        )
    }

    // MARK: - Embeddings

    public func processTokensForEmbedding(_ tokens: [Int32]) throws {
        guard let context = context else {
            throw CatalystError.engineNotInitialized
        }

        var embBatch = try LlamaBridge.createBatch(
            maxTokens: UInt32(tokens.count),
            embeddingSize: embeddingSize,
            numSequences: 1
        )
        defer { LlamaBridge.freeBatch(embBatch) }

        LlamaBridge.clearBatch(&embBatch)

        for (i, token) in tokens.enumerated() {
            LlamaBridge.addTokenToBatch(
                batch: &embBatch,
                token: token,
                position: Int32(i),
                sequenceId: 0,
                generateLogits: false
            )
        }

        try LlamaBridge.processBatch(context: context, batch: embBatch)
    }

    public func getEmbeddings() -> UnsafeMutablePointer<Float>? {
        guard let context = context else { return nil }
        return LlamaBridge.getEmbeddings(context: context)
    }

    // MARK: - State Persistence

    public var stateSize: Int {
        guard let context = context else { return 0 }
        return LlamaBridge.getStateSize(context)
    }

    public func saveState(to buffer: UnsafeMutablePointer<UInt8>, size: Int) -> Int {
        guard let context = context else { return 0 }
        return LlamaBridge.saveState(context, to: buffer, size: size)
    }

    public func loadState(from buffer: UnsafePointer<UInt8>, size: Int) -> Int {
        guard let context = context else { return 0 }
        return LlamaBridge.loadState(context, from: buffer, size: size)
    }
}
