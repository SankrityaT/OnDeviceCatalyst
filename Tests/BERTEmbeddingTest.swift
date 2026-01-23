//
//  BERTEmbeddingTest.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya on 2026-01-22.
//

import XCTest
@testable import OnDeviceCatalyst

final class BERTEmbeddingTest: XCTestCase {
    
    func testBGESmallEmbedding() async throws {
        // Test with actual BGE-small model
        let modelPath = "/Users/sankritya/Library/Application Support/SwiftMem/Models/bge-small-en-v1.5.gguf"
        
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("BGE model not found at \(modelPath)")
        }
        
        print("📦 Loading model from: \(modelPath)")
        
        let profile = try ModelProfile(filePath: modelPath)
        let config = PredictionConfig.balanced
        
        let llama = try LlamaInstance(profile: profile, config: config)
        
        print("🔄 Initializing model...")
        for await progress in llama.initialize() {
            print("📊 \(progress)")
            if case .ready = progress {
                break
            }
            if case .failed(let error) = progress {
                XCTFail("Initialization failed: \(error)")
                return
            }
        }
        
        print("✅ Model initialized")
        
        // Test 1: Short text
        print("\n🧪 Test 1: Short text")
        let shortText = "Hello world, this is a test."
        let shortEmbedding = try llama.embed(text: shortText)
        print("✅ Short embedding: dims=\(shortEmbedding.count), first 5 values: \(shortEmbedding.prefix(5))")
        XCTAssertEqual(shortEmbedding.count, 384, "BGE-small should produce 384-dim embeddings")
        
        // Test 2: Medium text
        print("\n🧪 Test 2: Medium text (~100 tokens)")
        let mediumText = String(repeating: "This is a test sentence. ", count: 20)
        let mediumEmbedding = try llama.embed(text: mediumText)
        print("✅ Medium embedding: dims=\(mediumEmbedding.count), first 5 values: \(mediumEmbedding.prefix(5))")
        XCTAssertEqual(mediumEmbedding.count, 384)
        
        // Test 3: Long text that requires chunking (>510 tokens)
        print("\n🧪 Test 3: Long text (>510 tokens, requires chunking)")
        let longText = String(repeating: "This is a test sentence that will be repeated many times to exceed the token limit. ", count: 100)
        let longEmbedding = try llama.embed(text: longText)
        print("✅ Long embedding: dims=\(longEmbedding.count), first 5 values: \(longEmbedding.prefix(5))")
        XCTAssertEqual(longEmbedding.count, 384, "Long text should also produce 384-dim embeddings via chunking")
        
        // Test 4: Multiple embeddings in sequence
        print("\n🧪 Test 4: Multiple embeddings in sequence")
        let texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        for (index, text) in texts.enumerated() {
            let embedding = try llama.embed(text: text)
            print("✅ Embedding \(index + 1): dims=\(embedding.count)")
            XCTAssertEqual(embedding.count, 384)
        }
        
        print("\n🎉 All tests passed!")
    }
}
