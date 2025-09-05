//
//  PromptFormatting.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/29/25.
//


//
//  PromptFormatter.swift
//  OnDeviceCatalyst
//
//  Architecture-specific prompt formatting for different model types
//

import Foundation

/// Protocol for formatting conversations into model-specific prompts
public protocol PromptFormatting {
    func formatPrompt(
        turns: [Turn],
        systemPrompt: String?,
        architecture: ModelArchitecture
    ) -> String
}

/// Standard prompt formatter supporting all major model architectures
public struct StandardPromptFormatter: PromptFormatting {
    
    public init() {}
    
    public func formatPrompt(
        turns: [Turn],
        systemPrompt: String?,
        architecture: ModelArchitecture
    ) -> String {
        let effectiveSystemPrompt = systemPrompt ?? "You are a helpful AI assistant."
        
        switch architecture {
        case .llama2, .codeLlama:
            return formatLlama2(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .llama3, .llama31:
            return formatLlama3(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .mistral, .mistralInstruct:
            return formatMistral(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .mixtral:
            return formatMixtral(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .phi3, .phi35:
            return formatPhi3(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .gemma, .gemma2:
            return formatGemma(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .qwen2, .qwen25, .codeQwen:
            return formatQwen(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .deepSeek, .deepSeekCoder:
            return formatDeepSeek(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .commandR:
            return formatCommandR(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .yi:
            return formatYi(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .openChat:
            return formatOpenChat(turns: turns, systemPrompt: effectiveSystemPrompt)
            
        case .unknown:
            print("Catalyst: Unknown architecture, using ChatML fallback")
            return formatChatML(turns: turns, systemPrompt: effectiveSystemPrompt)
        }
    }
    
    // MARK: - Llama 2 / Code Llama Format
    
    private func formatLlama2(turns: [Turn], systemPrompt: String) -> String {
        var prompt = "<<SYS>>\n\(systemPrompt)\n<</SYS>>\n\n"
        
        for turn in turns {
            switch turn.role {
            case .user:
                prompt += "[INST] \(turn.content) [/INST]"
            case .assistant:
                prompt += " \(turn.content) "
            case .system:
                // System turns are handled in the header
                break
            }
        }
        
        return prompt
    }
    
    // MARK: - Llama 3 / 3.1 Format
    
    private func formatLlama3(turns: [Turn], systemPrompt: String) -> String {
        var prompt = "<|begin_of_text|>"
        
        // Add system prompt
        prompt += "<|start_header_id|>system<|end_header_id|>\n\n\(systemPrompt)<|eot_id|>"
        
        // Add conversation turns
        for turn in turns {
            let roleString = turn.role.rawValue
            prompt += "<|start_header_id|>\(roleString)<|end_header_id|>\n\n\(turn.content)<|eot_id|>"
        }
        
        // Prepare for assistant response
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    }
    
    // MARK: - Mistral Format
    
    private func formatMistral(turns: [Turn], systemPrompt: String) -> String {
        var prompt = ""
        var isFirstUserTurn = true
        
        for turn in turns {
            switch turn.role {
            case .user:
                let userContent = isFirstUserTurn ? "\(systemPrompt)\n\(turn.content)" : turn.content
                prompt += "[INST] \(userContent) [/INST]"
                isFirstUserTurn = false
            case .assistant:
                prompt += " \(turn.content)</s>"
            case .system:
                // System handled in first user turn
                break
            }
        }
        
        return prompt
    }
    
    // MARK: - Mixtral Format
    
    private func formatMixtral(turns: [Turn], systemPrompt: String) -> String {
        // Mixtral uses similar format to Mistral but with slightly different handling
        return formatMistral(turns: turns, systemPrompt: systemPrompt)
    }
    
    // MARK: - Phi 3 Format
    
    private func formatPhi3(turns: [Turn], systemPrompt: String) -> String {
        var prompt = "<|system|>\n\(systemPrompt)<|end|>\n"
        
        for turn in turns {
            switch turn.role {
            case .user:
                prompt += "<|user|>\n\(turn.content)<|end|>\n"
            case .assistant:
                prompt += "<|assistant|>\n\(turn.content)<|end|>\n"
            case .system:
                // System handled in header
                break
            }
        }
        
        prompt += "<|assistant|>\n"
        return prompt
    }
    
    // MARK: - Gemma Format
    
    private func formatGemma(turns: [Turn], systemPrompt: String) -> String {
        var prompt = ""
        var isFirstUserTurn = true
        
        for turn in turns {
            switch turn.role {
            case .user:
                prompt += "<start_of_turn>user\n"
                if isFirstUserTurn {
                    prompt += "\(systemPrompt)\n\(turn.content)"
                    isFirstUserTurn = false
                } else {
                    prompt += turn.content
                }
                prompt += "<end_of_turn>\n"
            case .assistant:
                prompt += "<start_of_turn>model\n\(turn.content)<end_of_turn>\n"
            case .system:
                // System handled in first user turn
                break
            }
        }
        
        prompt += "<start_of_turn>model\n"
        return prompt
    }
    
    // MARK: - Qwen Format (ChatML-based)
    
    private func formatQwen(turns: [Turn], systemPrompt: String) -> String {
        return formatChatML(turns: turns, systemPrompt: systemPrompt)
    }
    
    // MARK: - DeepSeek Format
    
    private func formatDeepSeek(turns: [Turn], systemPrompt: String) -> String {
        var prompt = "User: \(systemPrompt)\n\n"
        
        for turn in turns {
            switch turn.role {
            case .user:
                prompt += "User: \(turn.content)\n\n"
            case .assistant:
                prompt += "Assistant: \(turn.content)\n\n"
            case .system:
                // System handled in header
                break
            }
        }
        
        prompt += "Assistant: "
        return prompt
    }
    
    // MARK: - Command R Format
    
    private func formatCommandR(turns: [Turn], systemPrompt: String) -> String {
        var prompt = "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\(systemPrompt)<|END_OF_TURN_TOKEN|>"
        
        for turn in turns {
            switch turn.role {
            case .user:
                prompt += "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\(turn.content)<|END_OF_TURN_TOKEN|>"
            case .assistant:
                prompt += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\(turn.content)<|END_OF_TURN_TOKEN|>"
            case .system:
                // System handled in header
                break
            }
        }
        
        prompt += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        return prompt
    }
    
    // MARK: - Yi Format (ChatML-based)
    
    private func formatYi(turns: [Turn], systemPrompt: String) -> String {
        return formatChatML(turns: turns, systemPrompt: systemPrompt)
    }
    
    // MARK: - OpenChat Format (ChatML-based)
    
    private func formatOpenChat(turns: [Turn], systemPrompt: String) -> String {
        return formatChatML(turns: turns, systemPrompt: systemPrompt)
    }
    
    // MARK: - ChatML Format (Fallback and base for several models)
    
    private func formatChatML(turns: [Turn], systemPrompt: String) -> String {
        var prompt = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
        
        for turn in turns {
            let roleString = turn.role.rawValue
            prompt += "<|im_start|>\(roleString)\n\(turn.content)<|im_end|>\n"
        }
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    }
}

// MARK: - Formatting Utilities

extension StandardPromptFormatter {
    
    /// Validates that a prompt is not empty and has reasonable length
    public func validatePrompt(_ prompt: String, maxLength: Int = 100_000) throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw CatalystError.configurationInvalid(
                parameter: "prompt",
                reason: "Generated prompt is empty"
            )
        }
        
        guard prompt.count <= maxLength else {
            throw CatalystError.configurationInvalid(
                parameter: "prompt",
                reason: "Generated prompt exceeds maximum length (\(prompt.count) > \(maxLength))"
            )
        }
    }
    
    /// Estimates token count for a formatted prompt
    public func estimateTokenCount(_ prompt: String) -> Int {
        // Rough approximation: 1 token per 4 characters for English text
        // This varies by tokenizer but gives a reasonable estimate
        return max(1, prompt.count / 4)
    }
    
    /// Creates a preview of the formatted prompt for debugging
    public func createPreview(_ prompt: String, maxLength: Int = 200) -> String {
        if prompt.count <= maxLength {
            return prompt
        }
        
        let truncated = String(prompt.prefix(maxLength))
        return truncated + "... (\(prompt.count - maxLength) more characters)"
    }
}

// MARK: - Architecture-Specific Utilities

extension ModelArchitecture {
    
    /// Returns the appropriate prompt formatter for this architecture
    public var promptFormatter: PromptFormatting {
        return StandardPromptFormatter()
    }
    
    /// Indicates if this architecture requires special beginning-of-sequence handling
    public var requiresBosToken: Bool {
        switch self {
        case .llama3, .llama31:
            return true
        default:
            return false
        }
    }
    
    /// Returns model-specific formatting notes for debugging
    public var formattingNotes: String {
        switch self {
        case .llama2, .codeLlama:
            return "Uses [INST]/[/INST] format with <<SYS>> system prompt wrapper"
        case .llama3, .llama31:
            return "Uses <|start_header_id|>/<|end_header_id|> format with <|begin_of_text|> prefix"
        case .mistral, .mistralInstruct:
            return "Uses [INST]/[/INST] format with system prompt in first user message"
        case .phi3, .phi35:
            return "Uses <|system|>/<|user|>/<|assistant|> format with <|end|> terminators"
        case .qwen2, .qwen25, .codeQwen, .yi, .openChat:
            return "Uses ChatML format with <|im_start|>/<|im_end|> tags"
        case .gemma, .gemma2:
            return "Uses <start_of_turn>/<end_of_turn> format with user/model roles"
        case .deepSeek, .deepSeekCoder:
            return "Uses simple User:/Assistant: format"
        case .commandR:
            return "Uses Command-R specific token format with special markers"
        case .unknown:
            return "Unknown architecture, falling back to ChatML format"
        default:
            return "Standard formatting for this architecture"
        }
    }
}
