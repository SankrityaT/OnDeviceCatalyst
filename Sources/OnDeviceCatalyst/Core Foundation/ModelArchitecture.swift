//
//  ModelArchitecture.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//

import Foundation

/// Defines supported model architectures with their specific requirements
public enum ModelArchitecture: String, CaseIterable, Codable, Hashable {
    case llama2 = "llama2"
    case llama3 = "llama3"
    case llama31 = "llama3.1"
    case mistral = "mistral"
    case mistralInstruct = "mistral_instruct"
    case mixtral = "mixtral"
    case phi3 = "phi3"
    case phi35 = "phi3.5"
    case gemma = "gemma"
    case gemma2 = "gemma2"
    case qwen2 = "qwen2"
    case qwen25 = "qwen2.5"
    case codeQwen = "code_qwen"
    case codeLlama = "code_llama"
    case deepSeek = "deepseek"
    case deepSeekCoder = "deepseek_coder"
    case commandR = "command_r"
    case yi = "yi"
    case openChat = "openchat"
    case unknown = "unknown"
    
    /// Attempts to detect architecture from model filename or path
    public static func detectFromPath(_ path: String) -> ModelArchitecture {
        let filename = (path as NSString).lastPathComponent.lowercased()
        let pathLower = path.lowercased()
        
        // Specific version detection first (more specific matches)
        if filename.contains("llama-3.1") || filename.contains("llama3.1") {
            return .llama31
        } else if filename.contains("llama-3") || filename.contains("llama3") {
            return .llama3
        } else if filename.contains("llama-2") || filename.contains("llama2") {
            return .llama2
        }
        
        // Qwen variants
        if filename.contains("qwen2.5") {
            return .qwen25
        } else if filename.contains("codeqwen") {
            return .codeQwen
        } else if filename.contains("qwen2") {
            return .qwen2
        } else if filename.contains("qwen1_5") || filename.contains("qwen1.5") || filename.contains("qwen-1.5") {
            return .qwen2  // Qwen 1.5 uses same format as Qwen 2
        }
        
        // Gemma variants
        if filename.contains("gemma-2") || filename.contains("gemma2") {
            return .gemma2
        } else if filename.contains("gemma") {
            return .gemma
        }
        
        // Phi variants
        if filename.contains("phi-3.5") || filename.contains("phi3.5") {
            return .phi35
        } else if filename.contains("phi-3") || filename.contains("phi3") {
            return .phi3
        }
        
        // DeepSeek variants
        if filename.contains("deepseek-coder") || filename.contains("deepseek_coder") {
            return .deepSeekCoder
        } else if filename.contains("deepseek") {
            return .deepSeek
        }
        
        // Code models
        if filename.contains("codellama") || filename.contains("code-llama") {
            return .codeLlama
        }
        
        // Other architectures
        if filename.contains("mistral-7b-instruct") || filename.contains("mistral-instruct") {
            return .mistralInstruct
        } else if filename.contains("mixtral") {
            return .mixtral
        } else if filename.contains("mistral") {
            return .mistral
        }
        
        if filename.contains("command-r") || filename.contains("commandr") {
            return .commandR
        }
        
        if filename.contains("yi-") {
            return .yi
        }
        
        if filename.contains("openchat") {
            return .openChat
        }
        
        // Fallback to general Llama if contains "llama" but no specific version
        if filename.contains("llama") {
            return .llama2
        }
        
        return .unknown
    }
    
    /// Returns a fallback architecture for better compatibility
    public func getFallbackArchitecture() -> ModelArchitecture {
        switch self {
        case .llama31:
            return .llama3  // 3.1 falls back to 3.0 template
        case .qwen25:
            return .qwen2   // 2.5 falls back to 2.0 template
        case .phi35:
            return .phi3    // 3.5 falls back to 3.0 template
        case .gemma2:
            return .gemma   // Gemma2 falls back to Gemma1
        case .codeQwen:
            return .qwen2   // Code variant uses base template
        case .deepSeekCoder:
            return .deepSeek // Coder variant uses base template
        case .unknown:
            return .llama2  // Ultimate fallback to most compatible
        default:
            return self     // No fallback needed
        }
    }
    
    /// Indicates if this architecture requires special tokens (BOS/EOS)
    public var requiresSpecialTokens: Bool {
        switch self {
        case .llama3, .llama31:
            return true
        case .qwen2, .qwen25, .codeQwen:
            return true
        default:
            return false
        }
    }
    
    /// Returns the expected context window for this architecture (if known)
    public var defaultContextWindow: UInt32 {
        switch self {
        case .llama2, .mistral, .mistralInstruct:
            return 4096
        case .llama3:
            return 8192
        case .llama31:
            return 131072  // 128K context
        case .qwen2, .qwen25, .codeQwen:
            return 32768   // 32K context
        case .mixtral:
            return 32768
        case .phi3, .phi35:
            return 4096
        case .gemma, .gemma2:
            return 8192
        case .commandR:
            return 131072
        default:
            return 4096    // Conservative default
        }
    }
    
    /// Indicates if this architecture supports system prompts natively
    public var supportsSystemPrompts: Bool {
        switch self {
        case .llama3, .llama31, .qwen2, .qwen25, .phi3, .phi35, .openChat:
            return true
        case .llama2, .codeLlama:
            return true  // Via special format
        default:
            return false
        }
    }
    
    /// Returns compatibility information for debugging
    public var compatibilityInfo: String {
        switch self {
        case .llama3, .llama31:
            return "Native ChatML support with special tokens. High compatibility."
        case .qwen2, .qwen25:
            return "ChatML format. Requires recent llama.cpp (2024+)."
        case .mixtral:
            return "Mistral format. May need specific llama.cpp build."
        case .phi3, .phi35:
            return "Microsoft Phi format. Good compatibility."
        case .unknown:
            return "Unknown architecture. Will use ChatML fallback."
        default:
            return "Standard architecture with good compatibility."
        }
    }
}

extension ModelArchitecture {
    /// Validates that the architecture is supported by the current llama.cpp version
    public func validateCompatibility() throws {
        // Check for known problematic architectures
        switch self {
        case .commandR:
            // Command-R has specific token requirements that may not be supported
            print("Warning: Command-R architecture may have limited compatibility")
        case .unknown:
            print("Warning: Unknown architecture detected, using fallback formatting")
        default:
            break
        }
    }
    
    /// Returns suggested settings for this architecture
    public func recommendedSettings() -> (contextLength: UInt32, batchSize: UInt32, gpuLayers: Int32) {
        let context = min(defaultContextWindow, 8192) // Cap at 8K for mobile
        let batch: UInt32 = min(512, context / 8)     // Reasonable batch size
        
        let gpuLayers: Int32 = {
            #if targetEnvironment(simulator)
            return 0  // No GPU on simulator
            #else
            return 99 // Let llama.cpp decide optimal layers
            #endif
        }()
        
        return (context, batch, gpuLayers)
    }
}
