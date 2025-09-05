//
//  DialogueRole.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/29/25.
//


//
//  DialogueTypes.swift
//  OnDeviceCatalyst
//
//  Core types for conversation handling and dialogue management
//

import Foundation

/// Represents the role of a participant in a conversation
public enum DialogueRole: String, Codable, CaseIterable, Hashable {
    case system = "system"
    case user = "user"
    case assistant = "assistant"
    
    /// Display name for UI purposes
    public var displayName: String {
        switch self {
        case .system:
            return "System"
        case .user:
            return "User"
        case .assistant:
            return "Assistant"
        }
    }
    
    /// Indicates if this role can be edited by users
    public var isUserEditable: Bool {
        switch self {
        case .user, .system:
            return true
        case .assistant:
            return false
        }
    }
}

/// Represents a single turn in a conversation
public struct Turn: Identifiable, Codable, Hashable {
    public let id: UUID
    public let role: DialogueRole
    public var content: String
    public let timestamp: Date
    public var metadata: TurnMetadata?
    
    public init(
        id: UUID = UUID(),
        role: DialogueRole,
        content: String,
        timestamp: Date = Date(),
        metadata: TurnMetadata? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata
    }
    
    /// Creates a user turn
    public static func user(_ content: String) -> Turn {
        Turn(role: .user, content: content)
    }
    
    /// Creates an assistant turn
    public static func assistant(_ content: String, metadata: TurnMetadata? = nil) -> Turn {
        Turn(role: .assistant, content: content, metadata: metadata)
    }
    
    /// Creates a system turn
    public static func system(_ content: String) -> Turn {
        Turn(role: .system, content: content)
    }
    
    /// Returns word count for the turn content
    public var wordCount: Int {
        content.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .count
    }
    
    /// Returns character count for the turn content
    public var characterCount: Int {
        content.count
    }
    
    /// Estimated token count (rough approximation)
    public var estimatedTokenCount: Int {
        // Rough estimate: 1 token per 4 characters for English text
        return max(1, content.count / 4)
    }
}

/// Metadata associated with a conversation turn
public struct TurnMetadata: Codable, Hashable {
    public var tokensGenerated: Int?
    public var generationTimeMs: Int64?
    public var tokensPerSecond: Double?
    public var completionReason: CompletionReason?
    public var modelUsed: String?
    public var temperature: Float?
    public var finishReason: String?
    
    public init(
        tokensGenerated: Int? = nil,
        generationTimeMs: Int64? = nil,
        tokensPerSecond: Double? = nil,
        completionReason: CompletionReason? = nil,
        modelUsed: String? = nil,
        temperature: Float? = nil,
        finishReason: String? = nil
    ) {
        self.tokensGenerated = tokensGenerated
        self.generationTimeMs = generationTimeMs
        self.tokensPerSecond = tokensPerSecond
        self.completionReason = completionReason
        self.modelUsed = modelUsed
        self.temperature = temperature
        self.finishReason = finishReason
    }
}

/// Represents a complete conversation with management utilities
public struct Conversation: Identifiable, Codable {
    public let id: UUID
    public var title: String
    public var turns: [Turn]
    public var systemPrompt: String?
    public let createdAt: Date
    public var updatedAt: Date
    public var metadata: ConversationMetadata
    
    public init(
        id: UUID = UUID(),
        title: String = "New Conversation",
        turns: [Turn] = [],
        systemPrompt: String? = nil,
        createdAt: Date = Date(),
        metadata: ConversationMetadata = ConversationMetadata()
    ) {
        self.id = id
        self.title = title
        self.turns = turns
        self.systemPrompt = systemPrompt
        self.createdAt = createdAt
        self.updatedAt = createdAt
        self.metadata = metadata
    }
    
    // MARK: - Conversation Management
    
    /// Adds a turn to the conversation
    public mutating func addTurn(_ turn: Turn) {
        turns.append(turn)
        updatedAt = Date()
        
        // Auto-generate title from first user turn if still using default
        if title == "New Conversation", turn.role == .user, !turn.content.isEmpty {
            title = generateTitle(from: turn.content)
        }
    }
    
    /// Removes a turn by ID
    public mutating func removeTurn(withId id: UUID) {
        turns.removeAll { $0.id == id }
        updatedAt = Date()
    }
    
    /// Updates a turn's content
    public mutating func updateTurn(withId id: UUID, content: String) {
        if let index = turns.firstIndex(where: { $0.id == id }) {
            turns[index].content = content
            updatedAt = Date()
        }
    }
    
    /// Clears all turns while preserving conversation metadata
    public mutating func clearTurns() {
        turns.removeAll()
        updatedAt = Date()
    }
    
    // MARK: - Context Management
    
    /// Returns turns that fit within the token limit
    public func turnsWithinLimit(tokenLimit: Int, excludeSystem: Bool = false) -> [Turn] {
        var totalTokens = 0
        var validTurns: [Turn] = []
        
        // Always include system prompt if present and not excluded
        if !excludeSystem, let systemPrompt = systemPrompt, !systemPrompt.isEmpty {
            let systemTurn = Turn.system(systemPrompt)
            totalTokens += systemTurn.estimatedTokenCount
            if totalTokens <= tokenLimit {
                validTurns.append(systemTurn)
            }
        }
        
        // Add turns from most recent, staying within token limit
        for turn in turns.reversed() {
            let turnTokens = turn.estimatedTokenCount
            if totalTokens + turnTokens <= tokenLimit {
                totalTokens += turnTokens
                validTurns.insert(turn, at: excludeSystem ? 0 : 1)
            } else {
                break
            }
        }
        
        return validTurns
    }
    
    /// Returns conversation turns optimized for a given context length
    public func optimizedTurns(contextLength: UInt32, reserveTokens: Int = 1000) -> [Turn] {
        let availableTokens = max(0, Int(contextLength) - reserveTokens)
        return turnsWithinLimit(tokenLimit: availableTokens)
    }
    
    // MARK: - Statistics
    
    /// Total estimated token count for the conversation
    public var estimatedTokenCount: Int {
        let systemTokens = systemPrompt?.count ?? 0 / 4
        let turnTokens = turns.reduce(0) { $0 + $1.estimatedTokenCount }
        return systemTokens + turnTokens
    }
    
    /// Number of user turns
    public var userTurnCount: Int {
        turns.filter { $0.role == .user }.count
    }
    
    /// Number of assistant turns
    public var assistantTurnCount: Int {
        turns.filter { $0.role == .assistant }.count
    }
    
    /// Duration of the conversation
    public var duration: TimeInterval {
        guard let firstTurn = turns.first, let lastTurn = turns.last else {
            return 0
        }
        return lastTurn.timestamp.timeIntervalSince(firstTurn.timestamp)
    }
    
    // MARK: - Private Helpers
    
    private func generateTitle(from content: String) -> String {
        let words = content.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .prefix(6)
        
        let title = words.joined(separator: " ")
        return title.isEmpty ? "New Conversation" : title
    }
}

/// Metadata for conversation tracking and analytics
public struct ConversationMetadata: Codable {
    public var modelUsed: String?
    public var totalTokensGenerated: Int
    public var totalGenerationTime: TimeInterval
    public var averageResponseTime: TimeInterval?
    public var tags: Set<String>
    public var isArchived: Bool
    public var isFavorite: Bool
    
    public init(
        modelUsed: String? = nil,
        totalTokensGenerated: Int = 0,
        totalGenerationTime: TimeInterval = 0,
        averageResponseTime: TimeInterval? = nil,
        tags: Set<String> = [],
        isArchived: Bool = false,
        isFavorite: Bool = false
    ) {
        self.modelUsed = modelUsed
        self.totalTokensGenerated = totalTokensGenerated
        self.totalGenerationTime = totalGenerationTime
        self.averageResponseTime = averageResponseTime
        self.tags = tags
        self.isArchived = isArchived
        self.isFavorite = isFavorite
    }
}

// MARK: - Collection Extensions

extension Array where Element == Turn {
    /// Returns only user and assistant turns (excludes system)
    public var dialogueTurns: [Turn] {
        filter { $0.role == .user || $0.role == .assistant }
    }
    
    /// Returns the most recent turn of a specific role
    public func lastTurn(of role: DialogueRole) -> Turn? {
        last { $0.role == role }
    }
    
    /// Estimated total token count for the turn array
    public var estimatedTokenCount: Int {
        reduce(0) { $0 + $1.estimatedTokenCount }
    }
    
    /// Returns turns within a token budget, keeping most recent
    public func withinTokenBudget(_ budget: Int) -> [Turn] {
        var totalTokens = 0
        var result: [Turn] = []
        
        for turn in reversed() {
            let turnTokens = turn.estimatedTokenCount
            if totalTokens + turnTokens <= budget {
                totalTokens += turnTokens
                result.insert(turn, at: 0)
            } else {
                break
            }
        }
        
        return result
    }
}