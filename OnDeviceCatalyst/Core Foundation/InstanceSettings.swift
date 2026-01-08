//
//  InstanceSettings.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//

import Foundation

/// Configuration settings for model instance initialization and execution
public struct InstanceSettings: Codable, Hashable {
    
    // MARK: - Context Configuration
    public var contextLength: UInt32
    public var batchSize: UInt32
    
    // MARK: - GPU/CPU Configuration  
    public var gpuLayers: Int32
    public var cpuThreads: Int32
    
    // MARK: - Memory Configuration
    public var enableMemoryMapping: Bool
    public var enableMemoryLocking: Bool
    
    // MARK: - Advanced Features
    public var useFlashAttention: Bool
    public var seed: UInt32
    
    /// Initialize with default values optimized for the current device
    public init(
        contextLength: UInt32 = 4096,
        batchSize: UInt32 = 512,
        gpuLayers: Int32 = DeviceOptimizer.recommendedGpuLayers(),
        cpuThreads: Int32 = DeviceOptimizer.recommendedCpuThreads(),
        enableMemoryMapping: Bool = true,
        enableMemoryLocking: Bool = false,
        useFlashAttention: Bool = false,
        seed: UInt32 = 0
    ) {
        self.contextLength = contextLength
        self.batchSize = batchSize
        self.gpuLayers = gpuLayers
        self.cpuThreads = cpuThreads
        self.enableMemoryMapping = enableMemoryMapping
        self.enableMemoryLocking = enableMemoryLocking
        self.useFlashAttention = useFlashAttention
        self.seed = seed == 0 ? UInt32.random(in: 1...UInt32.max) : seed
    }
    
    // MARK: - Preset Configurations
    
    /// Balanced settings suitable for most use cases
    public static var balanced: InstanceSettings {
        InstanceSettings()
    }
    
    /// iPhone 16 Pro Max optimized settings - 3B model with GPU acceleration
    public static var iphone16ProMax: InstanceSettings {
        return InstanceSettings(
            contextLength: 2048,        // Conservative for 3B model
            batchSize: 256,             // Smaller batch for stability
            gpuLayers: 25,              // Partial GPU for 3B model
            cpuThreads: 6,              // Leave cores for system
            enableMemoryMapping: true,
            enableMemoryLocking: false, // iOS safety
            useFlashAttention: true,
            seed: 0
        )
    }
    
    /// Coaching app settings with flexible context and batch sizes
    public static func coaching(
        maxContextTokens: Int = 6144,
        expectedPromptSize: Int = 1000
    ) -> InstanceSettings {
        let contextLength = min(maxContextTokens, max(2048, expectedPromptSize * 3))
        let batchSize = min(384, max(128, contextLength / 16))
        
        return InstanceSettings(
            contextLength: UInt32(contextLength),
            batchSize: UInt32(batchSize),
            gpuLayers: 99,
            cpuThreads: 8,
            enableMemoryMapping: true,
            enableMemoryLocking: false,
            useFlashAttention: true,
            seed: 0
        )
    }
    
    /// Memory-efficient settings for resource-constrained devices
    public static var memoryEfficient: InstanceSettings {
        InstanceSettings(
            contextLength: 1024,
            batchSize: 128,
            gpuLayers: 0, // CPU only
            enableMemoryLocking: false,
            useFlashAttention: false
        )
    }
    
    /// High-capacity settings for maximum context length
    public static var highCapacity: InstanceSettings {
        #if os(iOS)
        return InstanceSettings(
            contextLength: 6144,
            batchSize: 256,
            useFlashAttention: true
        )
        #else
        return InstanceSettings(
            contextLength: 16384,
            batchSize: 512,
            useFlashAttention: true
        )
        #endif
    }
    
    /// Embedding-optimized settings for semantic search models (e.g., bge-small)
    /// Optimized for fast embedding generation with minimal context requirements
    public static func embedding(outputDimensions: Int = 384) -> InstanceSettings {
        return InstanceSettings(
            contextLength: 512,         // Small context for embedding models
            batchSize: 128,             // Smaller batch for faster processing
            gpuLayers: 99,              // Full GPU acceleration
            cpuThreads: 4,              // Fewer threads needed
            enableMemoryMapping: true,
            enableMemoryLocking: false,
            useFlashAttention: false,   // Not needed for embeddings
            seed: 0
        )
    }
    
    // MARK: - Validation
    
    /// Validates that the settings are reasonable and compatible
    public func validate() throws {
        // Context length validation
        guard contextLength >= 256 else {
            throw CatalystError.configurationInvalid(
                parameter: "contextLength",
                reason: "Context length must be at least 256 tokens"
            )
        }
        
        guard contextLength <= 131072 else {
            throw CatalystError.configurationInvalid(
                parameter: "contextLength", 
                reason: "Context length cannot exceed 131,072 tokens (128K)"
            )
        }
        
        // Batch size validation
        guard batchSize >= 1 else {
            throw CatalystError.configurationInvalid(
                parameter: "batchSize",
                reason: "Batch size must be at least 1"
            )
        }
        
        guard batchSize <= contextLength else {
            throw CatalystError.configurationInvalid(
                parameter: "batchSize",
                reason: "Batch size cannot exceed context length"
            )
        }
        
        // GPU layers validation
        guard gpuLayers >= 0 else {
            throw CatalystError.configurationInvalid(
                parameter: "gpuLayers",
                reason: "GPU layers cannot be negative"
            )
        }
        
        // Thread validation
        guard cpuThreads >= 1 else {
            throw CatalystError.configurationInvalid(
                parameter: "cpuThreads",
                reason: "CPU threads must be at least 1"
            )
        }
        
        guard cpuThreads <= 32 else {
            throw CatalystError.configurationInvalid(
                parameter: "cpuThreads",
                reason: "CPU threads should not exceed 32"
            )
        }
        
        // Memory locking requires memory mapping
        if enableMemoryLocking && !enableMemoryMapping {
            throw CatalystError.configurationInvalid(
                parameter: "enableMemoryLocking",
                reason: "Memory locking requires memory mapping to be enabled"
            )
        }
    }
    
    /// Returns estimated memory usage in bytes
    public var estimatedMemoryUsage: UInt64 {
        // Rough estimate: context tokens * 4 bytes per token * 2 for KV cache
        let contextMemory = UInt64(contextLength) * 4 * 2
        
        // Batch processing memory
        let batchMemory = UInt64(batchSize) * 4 * 16 // rough estimate
        
        // Base model memory (varies by model, this is a conservative estimate)
        let baseMemory: UInt64 = 1024 * 1024 * 1024 // 1GB base
        
        return contextMemory + batchMemory + baseMemory
    }
    
    /// Returns a copy with optimizations for the given model architecture
    public func optimizedFor(_ architecture: ModelArchitecture) -> InstanceSettings {
        var optimized = self
        
        switch architecture {
        case .llama3, .llama31:
            optimized.useFlashAttention = true
            if contextLength < 4096 {
                optimized.contextLength = 4096
            }
            
        case .qwen2, .qwen25:
            optimized.useFlashAttention = true
            if contextLength < 8192 {
                optimized.contextLength = 8192
            }
            
        case .codeLlama, .codeQwen, .deepSeekCoder:
            // Code models benefit from larger context
            if contextLength < 4096 {
                optimized.contextLength = 4096
            }
            
        case .phi3, .phi35:
            // Phi models are efficient, can use higher settings
            optimized.useFlashAttention = true
            
        case .unknown:
            // Conservative settings for unknown architectures
            optimized.useFlashAttention = false
            if contextLength > 4096 {
                optimized.contextLength = 4096
            }
            
        default:
            break
        }
        
        return optimized
    }
}
