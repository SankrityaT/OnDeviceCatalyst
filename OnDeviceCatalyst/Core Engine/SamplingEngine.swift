//
//  SamplingEngine.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/29/25.
//


import Foundation
import llama

/// Advanced sampling engine implementing various token selection strategies
public class SamplingEngine {
    private let model: CModel
    private let context: CContext
    private let vocabularySize: Int32
    
    public init(model: CModel, context: CContext) {
        self.model = model
        self.context = context
        self.vocabularySize = LlamaBridge.getVocabularySize(model)
    }
    
    /// Main sampling method that applies all configured sampling techniques
    public func sampleToken(
        logits: UnsafeMutablePointer<Float>,
        config: PredictionConfig,
        recentTokens: [CToken] = []
    ) throws -> CToken {
        
        // Validate inputs
        try validateSamplingInputs(config: config, logits: logits)
        
        // Apply temperature early if using greedy sampling
        if config.temperature <= 0.001 {
            return sampleGreedy(logits: logits)
        }
        
        // Create working copy of logits
        var workingLogits = Array<Float>(
            UnsafeBufferPointer(start: logits, count: Int(vocabularySize))
        )
        
        // Apply repetition penalties
        if !recentTokens.isEmpty {
            applyRepetitionPenalties(
                logits: &workingLogits,
                recentTokens: recentTokens,
                config: config
            )
        }
        
        // Apply Mirostat sampling if enabled
        if config.mirostatMode > 0 {
            return try sampleMirostat(
                logits: workingLogits,
                config: config
            )
        }
        
        // Apply temperature scaling
        applyTemperature(logits: &workingLogits, temperature: config.temperature)
        
        // Convert to probabilities
        let probabilities = softmax(workingLogits)
        
        // Apply various sampling methods in sequence
        var candidates = createCandidates(from: probabilities)
        
        // Apply top-k filtering
        if config.topK > 0 {
            candidates = applyTopK(candidates: candidates, k: Int(config.topK))
        }
        
        // Apply top-p (nucleus) sampling
        if config.topP < 1.0 {
            candidates = applyTopP(candidates: candidates, p: config.topP)
        }
        
        // Apply min-p filtering
        if config.minP > 0.0 {
            candidates = applyMinP(candidates: candidates, minP: config.minP)
        }
        
        // Apply typical-p sampling
        if config.typicalP < 1.0 {
            candidates = applyTypicalP(candidates: candidates, typicalP: config.typicalP)
        }
        
        // Sample from remaining candidates
        return sampleFromCandidates(candidates)
    }
    
    // MARK: - Core Sampling Methods
    
    /// Greedy sampling - always pick highest probability token
    private func sampleGreedy(logits: UnsafeMutablePointer<Float>) -> CToken {
        var maxLogit: Float = -Float.infinity
        var bestToken: CToken = 0
        
        for i in 0..<Int(vocabularySize) {
            if logits[i] > maxLogit {
                maxLogit = logits[i]
                bestToken = CToken(i)
            }
        }
        
        return bestToken
    }
    
    /// Temperature scaling
    private func applyTemperature(logits: inout [Float], temperature: Float) {
        guard temperature > 0.0 && temperature != 1.0 else { return }
        
        for i in 0..<logits.count {
            logits[i] /= temperature
        }
    }
    
    /// Convert logits to probabilities using softmax
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0.0
        var probabilities = logits.map { exp($0 - maxLogit) }
        let sum = probabilities.reduce(0, +)
        
        guard sum > 0 else {
            // Fallback to uniform distribution
            let uniformProb = 1.0 / Float(logits.count)
            return Array(repeating: uniformProb, count: logits.count)
        }
        
        for i in 0..<probabilities.count {
            probabilities[i] /= sum
        }
        
        return probabilities
    }
    
    // MARK: - Sampling Filters
    
    /// Top-k sampling - keep only k highest probability tokens
    private func applyTopK(candidates: [TokenCandidate], k: Int) -> [TokenCandidate] {
        guard k > 0 && k < candidates.count else { return candidates }
        
        let sorted = candidates.sorted { $0.probability > $1.probability }
        return Array(sorted.prefix(k))
    }
    
    /// Top-p (nucleus) sampling - keep tokens with cumulative probability <= p
    private func applyTopP(candidates: [TokenCandidate], p: Float) -> [TokenCandidate] {
        guard p < 1.0 else { return candidates }
        
        let sorted = candidates.sorted { $0.probability > $1.probability }
        var cumulativeProb: Float = 0.0
        var result: [TokenCandidate] = []
        
        for candidate in sorted {
            cumulativeProb += candidate.probability
            result.append(candidate)
            
            if cumulativeProb >= p {
                break
            }
        }
        
        // Ensure we always have at least one candidate
        if result.isEmpty && !sorted.isEmpty {
            result.append(sorted[0])
        }
        
        return result
    }
    
    /// Min-p sampling - remove tokens with probability < (min_p * max_probability)
    private func applyMinP(candidates: [TokenCandidate], minP: Float) -> [TokenCandidate] {
        guard minP > 0.0 else { return candidates }
        
        let maxProb = candidates.max { $0.probability < $1.probability }?.probability ?? 0.0
        let threshold = minP * maxProb
        
        let filtered = candidates.filter { $0.probability >= threshold }
        return filtered.isEmpty ? [candidates.first!] : filtered
    }
    
    /// Typical-p sampling - locally typical sampling based on information theory
    private func applyTypicalP(candidates: [TokenCandidate], typicalP: Float) -> [TokenCandidate] {
       guard typicalP < 1.0 else { return candidates }
       
       // Calculate entropy and information content
       let entropy = candidates.reduce(0.0) { sum, candidate in
           let p = Double(candidate.probability)
           guard p > 0 else { return sum }
           return sum - (p * log2(p))
       }
        
        // Calculate absolute difference between token surprise and entropy
        let candidatesWithSurprise = candidates.map { candidate in
            let surprise = -log2(Double(candidate.probability))
            let diff = abs(surprise - entropy)
            return (candidate: candidate, surprise: Float(diff))
        }
        
        // Sort by how close they are to typical (smallest difference first)
        let sorted = candidatesWithSurprise.sorted { $0.surprise < $1.surprise }
        
        // Keep tokens until cumulative probability reaches typicalP
        var cumulativeProb: Float = 0.0
        var result: [TokenCandidate] = []
        
        for item in sorted {
            cumulativeProb += item.candidate.probability
            result.append(item.candidate)
            
            if cumulativeProb >= typicalP {
                break
            }
        }
        
        return result.isEmpty ? [candidates.first!] : result
    }
    
    // MARK: - Repetition Penalties
    
    /// Apply repetition penalties to recent tokens
    private func applyRepetitionPenalties(
        logits: inout [Float],
        recentTokens: [CToken],
        config: PredictionConfig
    ) {
        let penaltyRange = min(Int(config.repetitionPenaltyRange), recentTokens.count)
        let tokensToConsider = Array(recentTokens.suffix(penaltyRange))
        
        // Frequency counting for penalties
        var tokenCounts: [CToken: Int] = [:]
        for token in tokensToConsider {
            tokenCounts[token, default: 0] += 1
        }
        
        // Apply penalties
        for (token, count) in tokenCounts {
            let tokenIndex = Int(token)
            guard tokenIndex < logits.count else { continue }
            
            // Repetition penalty
            if config.repetitionPenalty != 1.0 {
                if logits[tokenIndex] > 0 {
                    logits[tokenIndex] /= config.repetitionPenalty
                } else {
                    logits[tokenIndex] *= config.repetitionPenalty
                }
            }
            
            // Frequency penalty - penalize based on frequency
            if config.frequencyPenalty != 0.0 {
                logits[tokenIndex] -= config.frequencyPenalty * Float(count)
            }
            
            // Presence penalty - penalize if token appeared at all
            if config.presencePenalty != 0.0 {
                logits[tokenIndex] -= config.presencePenalty
            }
        }
    }
    
    // MARK: - Mirostat Sampling
    
    private var mirostatTau: Float = 5.0
    
    /// Mirostat sampling for dynamic vocabulary control
    private func sampleMirostat(logits: [Float], config: PredictionConfig) throws -> CToken {
        // This is a simplified Mirostat implementation
        // Full Mirostat requires maintaining state across calls
        
        let probabilities = softmax(logits)
        let entropy = probabilities.reduce(0.0) { sum, p in
            p > 0 ? sum - (p * log2(p)) : sum
        }
        
        // Adjust tau based on entropy vs target
        let targetEntropy = config.mirostatTau
        let learningRate = config.mirostatEta
        mirostatTau += learningRate * (targetEntropy - entropy)
        mirostatTau = max(0.1, min(10.0, mirostatTau)) // Clamp tau
        
        // Use tau as effective temperature
        var adjustedLogits = logits
        applyTemperature(logits: &adjustedLogits, temperature: mirostatTau)
        
        let adjustedProbs = softmax(adjustedLogits)
        let candidates = createCandidates(from: adjustedProbs)
        
        return sampleFromCandidates(candidates)
    }
    
    // MARK: - Utilities
    
    /// Create candidate structure from probabilities
    private func createCandidates(from probabilities: [Float]) -> [TokenCandidate] {
        return probabilities.enumerated().map { index, probability in
            TokenCandidate(token: CToken(index), probability: probability)
        }
    }
    
    /// Sample from candidates using weighted random selection
    private func sampleFromCandidates(_ candidates: [TokenCandidate]) -> CToken {
        guard !candidates.isEmpty else { return 0 }
        
        // Renormalize probabilities
        let totalProb = candidates.reduce(0.0) { $0 + $1.probability }
        guard totalProb > 0 else { return candidates[0].token }
        
        let normalizedCandidates = candidates.map {
            TokenCandidate(token: $0.token, probability: $0.probability / totalProb)
        }
        
        // Weighted random selection
        let randomValue = Float.random(in: 0.0..<1.0)
        var cumulativeProb: Float = 0.0
        
        for candidate in normalizedCandidates {
            cumulativeProb += candidate.probability
            if randomValue <= cumulativeProb {
                return candidate.token
            }
        }
        
        // Fallback to last candidate
        return normalizedCandidates.last?.token ?? 0
    }
    
    /// Validate sampling inputs
    private func validateSamplingInputs(config: PredictionConfig, logits: UnsafeMutablePointer<Float>) throws {
        // Validate config was already done in PredictionConfig.validate()
        
        // Basic logits validation
        let firstLogit = logits[0]
        if firstLogit.isNaN || firstLogit.isInfinite {
            throw CatalystError.samplingFailed(
                method: "validation",
                reason: "Logits contain NaN or infinite values"
            )
        }
    }
}

// MARK: - Supporting Types

/// Represents a token candidate with its probability
private struct TokenCandidate {
    let token: CToken
    let probability: Float
}

// MARK: - Extensions

extension SamplingEngine {
    
    /// Quick sampling method using common settings
    public func quickSample(
        logits: UnsafeMutablePointer<Float>,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int32 = 40
    ) throws -> CToken {
        let config = PredictionConfig(
            temperature: temperature,
            topK: topK,
            topP: topP
        )
        
        return try sampleToken(logits: logits, config: config)
    }
    
    /// Get sampling statistics for debugging
    public func getSamplingStats(logits: UnsafeMutablePointer<Float>) -> SamplingStats {
        let logitsArray = Array(UnsafeBufferPointer(start: logits, count: Int(vocabularySize)))
        let probabilities = softmax(logitsArray)
        
        let maxProb = probabilities.max() ?? 0.0
        let entropy = probabilities.reduce(0.0) { sum, p in
            p > 0 ? sum - (p * log2(p)) : sum
        }
        
        let nonZeroCount = probabilities.filter { $0 > 0.001 }.count
        
        return SamplingStats(
            maxProbability: maxProb,
            entropy: entropy,
            effectiveVocabularySize: nonZeroCount,
            totalVocabularySize: Int(vocabularySize)
        )
    }
}

/// Statistics about the current sampling state
public struct SamplingStats {
    public let maxProbability: Float
    public let entropy: Float
    public let effectiveVocabularySize: Int
    public let totalVocabularySize: Int
    
    public var perplexity: Float {
        return pow(2, entropy)
    }
    
    public var vocabularyUtilization: Float {
        return Float(effectiveVocabularySize) / Float(totalVocabularySize)
    }
}
