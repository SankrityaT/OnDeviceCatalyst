//
//  EmbeddingTest.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya on 2026-01-22.
//

import XCTest
@testable import OnDeviceCatalyst

final class EmbeddingTest: XCTestCase {
    
    func testBGEEmbedding() async throws {
        // This test verifies that BGE-small embeddings work with llama_get_embeddings_seq
        
        // You need to provide a path to a downloaded BGE model
        let modelPath = "/path/to/bge-small-en-v1.5.gguf"
        
        // Skip if model doesn't exist
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("BGE model not found at \(modelPath)")
        }
        
        let profile = try ModelProfile(filePath: modelPath)
        let config = PredictionConfig.balanced
        
        let llama = try LlamaInstance(profile: profile, config: config)
        
        // Initialize model
        for await progress in llama.initialize() {
            print("Init: \(progress)")
            if case .ready = progress {
                break
            }
            if case .failed(let error) = progress {
                XCTFail("Initialization failed: \(error)")
                return
            }
        }
        
        // Test short text
        let shortText = "Hello world"
        let shortEmbedding = try llama.embed(text: shortText)
        XCTAssertEqual(shortEmbedding.count, 384, "BGE-small should produce 384-dim embeddings")
        
        // Test long text (>510 tokens) - should use chunking
        let longText = String(repeating: "This is a test sentence that will be repeated many times to exceed the token limit. ", count: 100)
        let longEmbedding = try llama.embed(text: longText)
        XCTAssertEqual(longEmbedding.count, 384, "Long text should also produce 384-dim embeddings")
        
        print("✅ Short embedding: \(shortEmbedding.prefix(5))")
        print("✅ Long embedding: \(longEmbedding.prefix(5))")
    }
}
