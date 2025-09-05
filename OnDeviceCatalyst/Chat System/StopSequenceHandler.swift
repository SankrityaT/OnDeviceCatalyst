//
//  StopSequenceHandler.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/28/25.
//


//
//  StopSequenceHandler.swift
//  OnDeviceCatalyst
//
//  Manages stop sequences and end-of-generation detection per model architecture
//

import Foundation

/// Manages stop sequences for different model architectures
public struct StopSequenceHandler {
    public let architecture: ModelArchitecture
    public var customStopSequences: [String]
    
    public init(architecture: ModelArchitecture, customStopSequences: [String] = []) {
        self.architecture = architecture
        self.customStopSequences = customStopSequences
    }
    
    /// Returns default stop sequences for the model architecture
    public var defaultStopSequences: [String] {
        switch architecture {
        case .llama2, .codeLlama:
            return ["</s>", "[INST]", "<<SYS>>"]
            
        case .llama3, .llama31:
            return [
                "<|eot_id|>",
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>system<|end_header_id|>",
                "<|end_of_text|>"
            ]
            
        case .mistral, .mistralInstruct:
            return ["</s>", "[INST]"]
            
        case .mixtral:
            return ["</s>", "[INST]"]
            
        case .phi3, .phi35:
            return ["<|end|>", "<|user|>", "<|system|>"]
            
        case .gemma, .gemma2:
            return [
                "<end_of_turn>",
                "<eos>",
                "<start_of_turn>user",
                "<start_of_turn>system"
            ]
            
        case .qwen2, .qwen25, .codeQwen:
            return [
                "<|im_end|>",
                "<|im_start|>user",
                "<|im_start|>system"
            ]
            
        case .deepSeek, .deepSeekCoder:
            return ["User:", "Assistant:"]
            
        case .commandR:
            return [
                "<|END_OF_TURN_TOKEN|>",
                "<|USER_TOKEN|>",
                "<|SYSTEM_TOKEN|>"
            ]
            
        case .yi:
            return [
                "<|im_end|>",
                "<|im_start|>user",
                "<|im_start|>system"
            ]
            
        case .openChat:
            return [
                "<|im_end|>",
                "<|im_start|>user",
                "<|im_start|>system"
            ]
            
        case .unknown:
            // Generic ChatML fallback
            return [
                "<|im_end|>",
                "<|im_start|>user",
                "<|im_start|>system"
            ]
        }
    }
    
    /// Returns all effective stop sequences (default + custom)
    public var allStopSequences: [String] {
        let combined = defaultStopSequences + customStopSequences
        return Array(Set(combined)) // Remove duplicates
    }
    
    /// Adds a custom stop sequence
    public mutating func addStopSequence(_ sequence: String) {
        if !sequence.isEmpty && !customStopSequences.contains(sequence) {
            customStopSequences.append(sequence)
        }
    }
    
    /// Removes a custom stop sequence
    public mutating func removeStopSequence(_ sequence: String) {
        customStopSequences.removeAll { $0 == sequence }
    }
    
    /// Clears all custom stop sequences
    public mutating func clearCustomStopSequences() {
        customStopSequences.removeAll()
    }
    
    // MARK: - Stop Sequence Detection
    
    /// Checks if the given text ends with any stop sequence
    public func detectStopSequence(in text: String) -> String? {
        for stopSequence in allStopSequences {
            if text.hasSuffix(stopSequence) {
                return stopSequence
            }
        }
        return nil
    }
    
    /// Checks if adding the new token would create a stop sequence
    public func wouldCreateStopSequence(currentText: String, newToken: String) -> String? {
        let combinedText = currentText + newToken
        return detectStopSequence(in: combinedText)
    }
    
    /// Returns the text with stop sequence removed (if found at the end)
    public func removeStopSequence(from text: String) -> (cleanedText: String, foundSequence: String?) {
        for stopSequence in allStopSequences.sorted(by: { $0.count > $1.count }) { // Check longer sequences first
            if text.hasSuffix(stopSequence) {
                let cleanedText = String(text.dropLast(stopSequence.count))
                return (cleanedText, stopSequence)
            }
        }
        return (text, nil)
    }
    
    /// Finds the portion of a token that should be yielded before hitting a stop sequence
    public func tokenPortionBeforeStop(currentText: String, newToken: String) -> (tokenPortion: String, stopSequence: String?) {
        let fullText = currentText + newToken
        
        for stopSequence in allStopSequences.sorted(by: { $0.count > $1.count }) {
            if let range = fullText.range(of: stopSequence, options: [.anchored, .backwards]) {
                // Find how much of the new token we can include
                let stopStart = range.lowerBound
                let currentTextEnd = fullText.index(fullText.startIndex, offsetBy: currentText.count)
                
                if stopStart <= currentTextEnd {
                    // Stop sequence starts in previous text, yield nothing
                    return ("", stopSequence)
                } else {
                    // Stop sequence starts in new token, yield partial token
                    let tokenStartInFullText = currentTextEnd
                    let portionEnd = stopStart
                    let portion = String(fullText[tokenStartInFullText..<portionEnd])
                    return (portion, stopSequence)
                }
            }
        }
        
        // No stop sequence found, return full token
        return (newToken, nil)
    }
    
    // MARK: - Validation
    
    /// Validates that stop sequences are reasonable for this architecture
    public func validateStopSequences() throws {
        let sequences = allStopSequences
        
        // Check for empty sequences
        if sequences.contains(where: { $0.isEmpty }) {
            throw CatalystError.configurationInvalid(
                parameter: "stopSequences",
                reason: "Stop sequences cannot be empty"
            )
        }
        
        // Check for very long sequences that might cause performance issues
        let maxLength = 100
        if let longSequence = sequences.first(where: { $0.count > maxLength }) {
            throw CatalystError.configurationInvalid(
                parameter: "stopSequences",
                reason: "Stop sequence '\(longSequence.prefix(20))...' is too long (\(longSequence.count) > \(maxLength))"
            )
        }
        
        // Warn about potentially problematic sequences
        let commonWords = ["the", "and", "or", "in", "on", "at", "to", "a", "an"]
        for sequence in sequences {
            if commonWords.contains(sequence.lowercased()) {
                print("Catalyst Warning: Stop sequence '\(sequence)' is a common word and may cause unexpected stops")
            }
        }
    }
    
    // MARK: - Debugging
    
    /// Returns information about stop sequences for debugging
    public var debugInfo: String {
        var info = "Stop Sequences for \(architecture.rawValue):\n"
        info += "Default: \(defaultStopSequences)\n"
        if !customStopSequences.isEmpty {
            info += "Custom: \(customStopSequences)\n"
        }
        info += "Total: \(allStopSequences.count) sequences"
        return info
    }
}

// MARK: - Architecture-Specific Utilities

extension ModelArchitecture {
    
    /// Creates a stop sequence handler configured for this architecture
    public func createStopSequenceHandler(customSequences: [String] = []) -> StopSequenceHandler {
        return StopSequenceHandler(architecture: self, customStopSequences: customSequences)
    }
    
    /// Returns critical stop sequences that should never be removed
    public var criticalStopSequences: [String] {
        switch self {
        case .llama3, .llama31:
            return ["<|eot_id|>"]
        case .llama2, .codeLlama:
            return ["</s>"]
        case .mistral, .mistralInstruct, .mixtral:
            return ["</s>"]
        case .phi3, .phi35:
            return ["<|end|>"]
        case .gemma, .gemma2:
            return ["<end_of_turn>"]
        case .qwen2, .qwen25, .codeQwen, .yi, .openChat:
            return ["<|im_end|>"]
        case .deepSeek, .deepSeekCoder:
            return [] // Uses simple format, no critical sequences
        case .commandR:
            return ["<|END_OF_TURN_TOKEN|>"]
        case .unknown:
            return ["<|im_end|>"] // ChatML fallback
        }
    }
}

// MARK: - Stop Sequence Analysis

extension StopSequenceHandler {
    
    /// Analyzes text for potential stop sequence issues
    public func analyzeText(_ text: String) -> StopSequenceAnalysis {
        var analysis = StopSequenceAnalysis()
        
        // Check for partial matches that might cause issues
        for sequence in allStopSequences {
            if text.contains(sequence) {
                analysis.containsStopSequences.append(sequence)
            }
            
            // Check for partial matches at the end
            for i in 1..<sequence.count {
                let partial = String(sequence.prefix(i))
                if text.hasSuffix(partial) {
                    analysis.partialMatches.append((partial, sequence))
                }
            }
        }
        
        return analysis
    }
}

/// Analysis results for stop sequence detection
public struct StopSequenceAnalysis {
    public var containsStopSequences: [String] = []
    public var partialMatches: [(partial: String, fullSequence: String)] = []
    
    public var hasIssues: Bool {
        return !containsStopSequences.isEmpty || !partialMatches.isEmpty
    }
    
    public var summary: String {
        var parts: [String] = []
        
        if !containsStopSequences.isEmpty {
            parts.append("Contains stop sequences: \(containsStopSequences)")
        }
        
        if !partialMatches.isEmpty {
            let partials = partialMatches.map { $0.partial }
            parts.append("Partial matches: \(partials)")
        }
        
        return parts.isEmpty ? "No stop sequence issues detected" : parts.joined(separator: "; ")
    }
}