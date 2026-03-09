//
//  InferenceBackend.swift
//  OnDeviceCatalyst
//
//  Protocol abstracting the inference engine so different backends
//  (llama.cpp, Metal, etc.) can be used interchangeably.
//

import Foundation

/// Selects which inference backend to use
public enum BackendType: String, Codable, Hashable {
    case llamaCpp
    case metal
}

/// Core abstraction for inference backends.
///
/// Both the llama.cpp wrapper and the custom Metal engine implement this protocol.
/// `LlamaInstance` drives the generation loop and calls these methods — the backend
/// only needs to provide the low-level primitives.
public protocol InferenceBackend: AnyObject {

    // MARK: - Lifecycle

    /// Load a GGUF model from disk.
    func loadModel(path: String, settings: InstanceSettings) throws

    /// Create an inference context (KV cache, batch buffers, etc.).
    func createContext(settings: InstanceSettings) throws

    /// Run a short warmup pass to prime GPU caches.
    func warmup() throws

    /// Release all resources.
    func shutdown()

    // MARK: - Model Metadata

    /// Number of tokens in the vocabulary.
    var vocabularySize: Int32 { get }

    /// Hidden dimension (embedding size).
    var embeddingSize: Int32 { get }

    /// Returns `true` if the token signals end-of-generation.
    func isEndOfGeneration(token: Int32) -> Bool

    // MARK: - Tokenization

    /// Convert text to token IDs.
    func tokenize(text: String, addBos: Bool, parseSpecial: Bool) throws -> [Int32]

    /// Convert a single token ID back to text.
    func detokenize(token: Int32) -> String

    // MARK: - Prompt Processing

    /// Process a batch of prompt tokens.
    ///
    /// - Parameters:
    ///   - tokens: Token IDs to process.
    ///   - positions: Absolute position for each token.
    ///   - generateLogitsAtLast: If `true`, generate logits for the last token in the batch.
    func processTokens(_ tokens: [Int32], positions: [Int32], generateLogitsAtLast: Bool) throws

    // MARK: - Decode Step

    /// Process a single token and generate logits (the decode step).
    func decodeToken(_ token: Int32, position: Int32) throws

    // MARK: - Logits

    /// Pointer to the vocab-sized float array of logits from the most recent decode.
    /// Valid until the next `processTokens` or `decodeToken` call.
    func getLogits() -> UnsafeMutablePointer<Float>?

    // MARK: - KV Cache

    /// Clear the entire KV cache.
    func clearKVCache()

    /// Remove KV cache entries for positions `[startPos, endPos)` in a sequence.
    func removeKVCacheTokens(sequenceId: Int32, startPos: Int32, endPos: Int32)

    // MARK: - Embeddings

    /// Process tokens specifically for embedding extraction.
    func processTokensForEmbedding(_ tokens: [Int32]) throws

    /// Pointer to the embedding vector from the most recent embedding pass.
    func getEmbeddings() -> UnsafeMutablePointer<Float>?

    // MARK: - State Persistence

    /// Size in bytes needed to serialize the current state.
    var stateSize: Int { get }

    /// Serialize state into buffer. Returns bytes written.
    func saveState(to buffer: UnsafeMutablePointer<UInt8>, size: Int) -> Int

    /// Restore state from buffer. Returns bytes read.
    func loadState(from buffer: UnsafePointer<UInt8>, size: Int) -> Int
}
