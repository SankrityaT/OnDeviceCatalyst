//
//  LlamaSamplerChain.swift
//  OnDeviceCatalyst
//
//  Native llama.cpp sampler chain wrapper for GPU-accelerated sampling
//

import Foundation

/// Native sampler chain using llama.cpp's built-in sampler APIs
/// Provides GPU-accelerated sampling with proper state management
public class LlamaSamplerChain {

    private var chain: CSampler?
    private let config: PredictionConfig
    private let vocabSize: Int32
    private let seed: UInt32

    /// Whether the chain has been initialized
    public var isInitialized: Bool {
        return chain != nil
    }

    /// Initialize a sampler chain from prediction config
    public init(config: PredictionConfig, vocabSize: Int32, seed: UInt32? = nil) {
        self.config = config
        self.vocabSize = vocabSize
        self.seed = seed ?? UInt32.random(in: 0...UInt32.max)
        self.chain = nil
    }

    deinit {
        cleanup()
    }

    // MARK: - Chain Setup

    /// Build the sampler chain based on config
    public func build() throws {
        cleanup()

        guard let newChain = LlamaBridge.createSamplerChain(noPerf: false) else {
            throw CatalystError.samplingFailed(method: "chain_init", reason: "Failed to create sampler chain")
        }

        // Add samplers in order based on config
        try addSamplersToChain(newChain)

        self.chain = newChain
        print("LlamaSamplerChain: Built chain with \(config.samplingStrategy)")
    }

    private func addSamplersToChain(_ chain: CSampler) throws {
        // Add repetition penalty first (operates on logits)
        if config.repetitionPenalty > 1.0 || config.frequencyPenalty > 0 || config.presencePenalty > 0 {
            if let penalties = LlamaBridge.createPenaltiesSampler(
                lastN: config.repetitionPenaltyRange,
                repeatPenalty: config.repetitionPenalty,
                freqPenalty: config.frequencyPenalty,
                presencePenalty: config.presencePenalty
            ) {
                LlamaBridge.addSamplerToChain(chain, sampler: penalties)
            }
        }

        // Determine sampling mode based on config
        if config.temperature == 0.0 {
            // Greedy sampling
            try addGreedySamplers(chain)
        } else if config.mirostatMode == 1 {
            // Mirostat v1
            try addMirostatSampler(chain)
        } else if config.mirostatMode == 2 {
            // Mirostat v2
            try addMirostatV2Sampler(chain)
        } else {
            // Standard combined sampling
            try addCombinedSamplers(chain)
        }
    }

    private func addGreedySamplers(_ chain: CSampler) throws {
        guard let greedy = LlamaBridge.createGreedySampler() else {
            throw CatalystError.samplingFailed(method: "greedy", reason: "Failed to create greedy sampler")
        }
        LlamaBridge.addSamplerToChain(chain, sampler: greedy)
    }

    private func addMirostatSampler(_ chain: CSampler) throws {
        guard let mirostat = LlamaBridge.createMirostatSampler(
            nVocab: vocabSize,
            seed: seed,
            tau: config.mirostatTau,
            eta: config.mirostatEta,
            m: 100
        ) else {
            throw CatalystError.samplingFailed(method: "mirostat", reason: "Failed to create Mirostat sampler")
        }
        LlamaBridge.addSamplerToChain(chain, sampler: mirostat)
    }

    private func addMirostatV2Sampler(_ chain: CSampler) throws {
        guard let mirostat = LlamaBridge.createMirostatV2Sampler(
            seed: seed,
            tau: config.mirostatTau,
            eta: config.mirostatEta
        ) else {
            throw CatalystError.samplingFailed(method: "mirostat_v2", reason: "Failed to create Mirostat V2 sampler")
        }
        LlamaBridge.addSamplerToChain(chain, sampler: mirostat)
    }

    private func addCombinedSamplers(_ chain: CSampler) throws {
        // Temperature
        if config.temperature > 0 {
            if let temp = LlamaBridge.createTempSampler(temp: config.temperature) {
                LlamaBridge.addSamplerToChain(chain, sampler: temp)
            }
        }

        // Top-K (if enabled, i.e., > 0)
        if config.topK > 0 {
            if let topK = LlamaBridge.createTopKSampler(k: config.topK) {
                LlamaBridge.addSamplerToChain(chain, sampler: topK)
            }
        }

        // Top-P (if < 1.0)
        if config.topP < 1.0 {
            if let topP = LlamaBridge.createTopPSampler(p: config.topP, minKeep: 1) {
                LlamaBridge.addSamplerToChain(chain, sampler: topP)
            }
        }

        // Min-P (if > 0)
        if config.minP > 0 {
            if let minP = LlamaBridge.createMinPSampler(p: config.minP, minKeep: 1) {
                LlamaBridge.addSamplerToChain(chain, sampler: minP)
            }
        }

        // Typical-P (if < 1.0)
        if config.typicalP < 1.0 {
            if let typical = LlamaBridge.createTypicalSampler(p: config.typicalP, minKeep: 1) {
                LlamaBridge.addSamplerToChain(chain, sampler: typical)
            }
        }

        // Distribution sampler for final selection
        if let dist = LlamaBridge.createDistSampler(seed: seed) {
            LlamaBridge.addSamplerToChain(chain, sampler: dist)
        }
    }

    // MARK: - Sampling

    /// Sample a token using the chain
    public func sample(context: CContext) throws -> CToken {
        guard let chain = chain else {
            throw CatalystError.engineNotInitialized
        }

        return LlamaBridge.sampleWithChain(chain, context: context, batchIndex: -1)
    }

    /// Accept a token (updates sampler state for repetition tracking, etc.)
    public func accept(token: CToken) {
        guard let chain = chain else { return }
        LlamaBridge.acceptToken(chain, token: token)
    }

    /// Reset the sampler chain state
    public func reset() {
        guard let chain = chain else { return }
        LlamaBridge.resetSampler(chain)
    }

    /// Cleanup and free resources
    public func cleanup() {
        if let chain = chain {
            LlamaBridge.freeSampler(chain)
            self.chain = nil
        }
    }
}
