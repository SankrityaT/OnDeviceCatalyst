//
//  ModelArchitecture.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//

import Foundation

/// Defines supported model architectures with their specific requirements
/// Updated for llama.cpp b7870 with support for latest model architectures
public enum ModelArchitecture: String, CaseIterable, Codable, Hashable {
    // Llama family
    case llama2 = "llama2"
    case llama3 = "llama3"
    case llama31 = "llama3.1"
    case llama32 = "llama3.2"        // NEW: Llama 3.2 (1B, 3B models)
    case llama33 = "llama3.3"        // NEW: Llama 3.3 (70B)

    // Mistral family
    case mistral = "mistral"
    case mistralInstruct = "mistral_instruct"
    case mixtral = "mixtral"
    case mistralSmall = "mistral_small"  // NEW: Mistral Small 24B

    // Microsoft Phi family
    case phi3 = "phi3"
    case phi35 = "phi3.5"
    case phi4 = "phi4"               // NEW: Phi 4

    // Google Gemma family
    case gemma = "gemma"
    case gemma2 = "gemma2"
    case gemma3 = "gemma3"           // NEW: Gemma 3

    // Alibaba Qwen family
    case qwen2 = "qwen2"
    case qwen25 = "qwen2.5"
    case qwen3 = "qwen3"             // NEW: Qwen 3
    case codeQwen = "code_qwen"
    case qwenVL = "qwen_vl"          // NEW: Qwen Vision-Language

    // DeepSeek family
    case deepSeek = "deepseek"
    case deepSeekCoder = "deepseek_coder"
    case deepSeekV3 = "deepseek_v3"  // NEW: DeepSeek V3

    // Code models
    case codeLlama = "code_llama"
    case starcoder = "starcoder"     // NEW: StarCoder/StarCoder2

    // Other architectures
    case commandR = "command_r"
    case yi = "yi"
    case openChat = "openchat"
    case internLM = "internlm"       // NEW: InternLM
    case olmo = "olmo"               // NEW: OLMo

    // Recurrent architectures (NEW)
    case mamba = "mamba"             // NEW: Mamba state-space model
    case rwkv = "rwkv"               // NEW: RWKV recurrent

    // Embedding/encoder models
    case bert = "bert"               // BERT-based embedding models (bge-small, etc.)
    case nomic = "nomic"             // NEW: Nomic Embed

    case unknown = "unknown"
    
    /// Attempts to detect architecture from model filename or path
    public static func detectFromPath(_ path: String) -> ModelArchitecture {
        let filename = (path as NSString).lastPathComponent.lowercased()

        // Llama variants (most specific first)
        if filename.contains("llama-3.3") || filename.contains("llama3.3") {
            return .llama33
        } else if filename.contains("llama-3.2") || filename.contains("llama3.2") {
            return .llama32
        } else if filename.contains("llama-3.1") || filename.contains("llama3.1") {
            return .llama31
        } else if filename.contains("llama-3") || filename.contains("llama3") {
            return .llama3
        } else if filename.contains("llama-2") || filename.contains("llama2") {
            return .llama2
        }

        // Qwen variants (most specific first)
        if filename.contains("qwen3") || filename.contains("qwen-3") {
            return .qwen3
        } else if filename.contains("qwen2.5") || filename.contains("qwen-2.5") {
            return .qwen25
        } else if filename.contains("qwen-vl") || filename.contains("qwenvl") {
            return .qwenVL
        } else if filename.contains("codeqwen") {
            return .codeQwen
        } else if filename.contains("qwen2") || filename.contains("qwen-2") {
            return .qwen2
        } else if filename.contains("qwen1_5") || filename.contains("qwen1.5") || filename.contains("qwen-1.5") {
            return .qwen2
        }

        // Gemma variants
        if filename.contains("gemma-3") || filename.contains("gemma3") {
            return .gemma3
        } else if filename.contains("gemma-2") || filename.contains("gemma2") {
            return .gemma2
        } else if filename.contains("gemma") {
            return .gemma
        }

        // Phi variants
        if filename.contains("phi-4") || filename.contains("phi4") {
            return .phi4
        } else if filename.contains("phi-3.5") || filename.contains("phi3.5") {
            return .phi35
        } else if filename.contains("phi-3") || filename.contains("phi3") {
            return .phi3
        }

        // DeepSeek variants
        if filename.contains("deepseek-v3") || filename.contains("deepseekv3") || filename.contains("deepseek_v3") {
            return .deepSeekV3
        } else if filename.contains("deepseek-coder") || filename.contains("deepseek_coder") {
            return .deepSeekCoder
        } else if filename.contains("deepseek") {
            return .deepSeek
        }

        // Code models
        if filename.contains("codellama") || filename.contains("code-llama") {
            return .codeLlama
        }
        if filename.contains("starcoder") {
            return .starcoder
        }

        // Mistral variants
        if filename.contains("mistral-small") {
            return .mistralSmall
        } else if filename.contains("mistral-7b-instruct") || filename.contains("mistral-instruct") {
            return .mistralInstruct
        } else if filename.contains("mixtral") {
            return .mixtral
        } else if filename.contains("mistral") {
            return .mistral
        }

        // Other architectures
        if filename.contains("command-r") || filename.contains("commandr") {
            return .commandR
        }
        if filename.contains("yi-") {
            return .yi
        }
        if filename.contains("openchat") {
            return .openChat
        }
        if filename.contains("internlm") {
            return .internLM
        }
        if filename.contains("olmo") {
            return .olmo
        }

        // Recurrent models
        if filename.contains("mamba") {
            return .mamba
        }
        if filename.contains("rwkv") {
            return .rwkv
        }

        // Embedding models
        if filename.contains("nomic-embed") || filename.contains("nomic_embed") {
            return .nomic
        }
        if filename.contains("bge-") || filename.contains("bert") || filename.contains("e5-") || filename.contains("gte-") {
            return .bert
        }

        // Fallback to general Llama if contains "llama"
        if filename.contains("llama") {
            return .llama2
        }

        return .unknown
    }
    
    /// Returns a fallback architecture for better compatibility
    public func getFallbackArchitecture() -> ModelArchitecture {
        switch self {
        case .llama33:
            return .llama31 // 3.3 falls back to 3.1
        case .llama32:
            return .llama3  // 3.2 falls back to 3.0
        case .llama31:
            return .llama3  // 3.1 falls back to 3.0 template
        case .qwen3:
            return .qwen25  // Qwen 3 falls back to 2.5
        case .qwen25:
            return .qwen2   // 2.5 falls back to 2.0 template
        case .phi4:
            return .phi35   // Phi 4 falls back to 3.5
        case .phi35:
            return .phi3    // 3.5 falls back to 3.0 template
        case .gemma3:
            return .gemma2  // Gemma 3 falls back to Gemma 2
        case .gemma2:
            return .gemma   // Gemma2 falls back to Gemma1
        case .codeQwen, .qwenVL:
            return .qwen2   // Qwen variants use base template
        case .deepSeekV3:
            return .deepSeek // V3 falls back to base
        case .deepSeekCoder:
            return .deepSeek // Coder variant uses base template
        case .starcoder:
            return .codeLlama // StarCoder falls back to CodeLlama
        case .mistralSmall:
            return .mistral // Small falls back to base Mistral
        case .internLM, .olmo:
            return .llama3  // Use Llama 3 format
        case .mamba, .rwkv:
            return .llama2  // Recurrent models fall back to basic format
        case .nomic:
            return .bert    // Nomic falls back to BERT
        case .unknown:
            return .llama2  // Ultimate fallback to most compatible
        default:
            return self     // No fallback needed
        }
    }
    
    /// Indicates if this architecture requires special tokens (BOS/EOS)
    public var requiresSpecialTokens: Bool {
        switch self {
        case .llama3, .llama31, .llama32, .llama33:
            return true
        case .qwen2, .qwen25, .qwen3, .codeQwen, .qwenVL:
            return true
        case .phi4:
            return true
        case .bert, .nomic:
            return false  // BERT uses [CLS] and [SEP] handled by tokenizer
        default:
            return false
        }
    }

    /// Indicates if this is an encoder-only model (for embeddings, not generation)
    public var isEncoderOnly: Bool {
        switch self {
        case .bert, .nomic:
            return true
        default:
            return false
        }
    }

    /// Indicates if this is a recurrent (non-transformer) architecture
    public var isRecurrent: Bool {
        switch self {
        case .mamba, .rwkv:
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
        case .llama31, .llama33:
            return 131072  // 128K context
        case .llama32:
            return 131072  // 128K context
        case .qwen2, .qwen25, .qwen3, .codeQwen:
            return 32768   // 32K context (Qwen 2.5 supports up to 128K)
        case .qwenVL:
            return 8192    // Vision models typically use less context
        case .mixtral, .mistralSmall:
            return 32768
        case .phi3, .phi35:
            return 4096
        case .phi4:
            return 16384   // Phi 4 supports 16K
        case .gemma, .gemma2, .gemma3:
            return 8192
        case .deepSeekV3:
            return 65536   // DeepSeek V3 supports 64K
        case .commandR:
            return 131072
        case .starcoder:
            return 8192
        case .mamba, .rwkv:
            return 16384   // Recurrent models can handle long context efficiently
        default:
            return 4096    // Conservative default
        }
    }

    /// Indicates if this architecture supports system prompts natively
    public var supportsSystemPrompts: Bool {
        switch self {
        case .llama3, .llama31, .llama32, .llama33:
            return true
        case .qwen2, .qwen25, .qwen3, .qwenVL:
            return true
        case .phi3, .phi35, .phi4:
            return true
        case .gemma2, .gemma3:
            return true
        case .deepSeekV3:
            return true
        case .openChat, .internLM:
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
