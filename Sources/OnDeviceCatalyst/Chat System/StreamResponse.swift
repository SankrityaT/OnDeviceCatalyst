//
//  StreamChunk.swift
//  OnDeviceCatalyst
//
//  Created by Sankritya Thakur on 8/29/25.
//

import Foundation

/// Individual streaming response chunk with content and status
public struct StreamChunk: Codable {
    public let content: String
    public let isComplete: Bool
    public let metadata: ChunkMetadata?
    
    public init(content: String, isComplete: Bool = false, metadata: ChunkMetadata? = nil) {
        self.content = content
        self.isComplete = isComplete
        self.metadata = metadata
    }
    
    /// Creates a content chunk
    public static func content(_ text: String) -> StreamChunk {
        StreamChunk(content: text, isComplete: false)
    }
    
    /// Creates a completion chunk with reason
    public static func completion(reason: CompletionReason, metadata: ResponseMetadata? = nil) -> StreamChunk {
        let chunkMetadata = ChunkMetadata(
            completionReason: reason,
            responseMetadata: metadata
        )
        return StreamChunk(content: "", isComplete: true, metadata: chunkMetadata)
    }

    /// Creates a completion chunk with tool calls
    public static func completionWithToolCalls(
        reason: CompletionReason,
        toolCalls: [CatalystToolCall],
        metadata: ResponseMetadata? = nil
    ) -> StreamChunk {
        let chunkMetadata = ChunkMetadata(
            completionReason: reason,
            responseMetadata: metadata,
            toolCalls: toolCalls
        )
        return StreamChunk(content: "", isComplete: true, metadata: chunkMetadata)
    }

    /// Any tool calls in this chunk
    public var toolCalls: [CatalystToolCall] {
        return metadata?.toolCalls ?? []
    }
}

/// Metadata for individual stream chunks
public struct ChunkMetadata: Codable {
    public let timestamp: Date
    public let chunkIndex: Int?
    public let completionReason: CompletionReason?
    public let responseMetadata: ResponseMetadata?
    public let toolCalls: [CatalystToolCall]?

    public init(
        timestamp: Date = Date(),
        chunkIndex: Int? = nil,
        completionReason: CompletionReason? = nil,
        responseMetadata: ResponseMetadata? = nil,
        toolCalls: [CatalystToolCall]? = nil
    ) {
        self.timestamp = timestamp
        self.chunkIndex = chunkIndex
        self.completionReason = completionReason
        self.responseMetadata = responseMetadata
        self.toolCalls = toolCalls
    }
}

/// Complete streaming response with accumulated content and metadata
public struct StreamingResponse {
    public private(set) var chunks: [StreamChunk] = []
    public private(set) var accumulatedContent: String = ""
    public private(set) var isComplete: Bool = false
    public private(set) var completionReason: CompletionReason?
    public private(set) var metadata: ResponseMetadata?
    
    public init() {}
    
    /// Adds a chunk to the response
    public mutating func addChunk(_ chunk: StreamChunk) {
        chunks.append(chunk)
        accumulatedContent += chunk.content
        
        if chunk.isComplete {
            isComplete = true
            completionReason = chunk.metadata?.completionReason
            metadata = chunk.metadata?.responseMetadata
        }
    }
    
    /// Current word count
    public var wordCount: Int {
        accumulatedContent
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .count
    }
    
    /// Estimated token count
    public var estimatedTokenCount: Int {
        max(1, accumulatedContent.count / 4)
    }
    
    /// Performance summary if available
    public var performanceSummary: String? {
        guard let metadata = metadata else { return nil }
        return metadata.usageSummary
    }

    /// Parse tool calls from the accumulated content
    /// Returns the text with tool calls removed, and the parsed tool calls
    public func parseToolCalls() -> (text: String, toolCalls: [CatalystToolCall]) {
        return ToolCallParser.parse(from: accumulatedContent)
    }

    /// All tool calls from completion chunks
    public var toolCalls: [CatalystToolCall] {
        return chunks.flatMap { $0.toolCalls }
    }
}

// MARK: - Stream Processing Utilities

/// Utility for processing streaming responses
public class StreamProcessor {
    private var buffer: String = ""
    private var chunkIndex: Int = 0
    private let stopHandler: StopSequenceHandler
    
    public init(stopHandler: StopSequenceHandler) {
        self.stopHandler = stopHandler
    }
    
    /// Processes a raw token into stream chunks, handling stop sequences
    public func processToken(_ token: String) -> [StreamChunk] {
        var chunks: [StreamChunk] = []
        buffer += token
        
        // Check for stop sequences
        let (cleanToken, stopSequence) = stopHandler.tokenPortionBeforeStop(
            currentText: buffer.dropLast(token.count).description,
            newToken: token
        )
        
        if !cleanToken.isEmpty {
            let chunk = StreamChunk(
                content: cleanToken,
                isComplete: false,
                metadata: ChunkMetadata(chunkIndex: chunkIndex)
            )
            chunks.append(chunk)
            chunkIndex += 1
        }
        
        if let stopSeq = stopSequence {
            let completionChunk = StreamChunk.completion(
                reason: .stopSequenceFound(stopSeq)
            )
            chunks.append(completionChunk)
        }
        
        return chunks
    }
    
    /// Finalizes stream processing
    public func finalize(reason: CompletionReason, metadata: ResponseMetadata? = nil) -> StreamChunk {
        return StreamChunk.completion(reason: reason, metadata: metadata)
    }
    
    /// Resets the processor for a new stream
    public func reset() {
        buffer = ""
        chunkIndex = 0
    }
}

// MARK: - Stream Collection

/// Collects and manages multiple streaming responses
public class StreamCollection {
    private var responses: [UUID: StreamingResponse] = [:]
    private let maxResponses: Int
    
    public init(maxResponses: Int = 10) {
        self.maxResponses = maxResponses
    }
    
    /// Starts a new streaming response
    public func startResponse() -> UUID {
        let id = UUID()
        responses[id] = StreamingResponse()
        
        // Cleanup old responses if needed
        if responses.count > maxResponses {
            let oldestId = responses.keys.min { a, b in
                (responses[a]?.chunks.first?.metadata?.timestamp ?? Date.distantPast) <
                (responses[b]?.chunks.first?.metadata?.timestamp ?? Date.distantPast)
            }
            if let oldId = oldestId {
                responses.removeValue(forKey: oldId)
            }
        }
        
        return id
    }
    
    /// Adds a chunk to a streaming response
    public func addChunk(_ chunk: StreamChunk, to responseId: UUID) {
        responses[responseId]?.addChunk(chunk)
    }
    
    /// Gets a streaming response
    public func getResponse(_ responseId: UUID) -> StreamingResponse? {
        return responses[responseId]
    }
    
    /// Removes a completed response
    public func removeResponse(_ responseId: UUID) {
        responses.removeValue(forKey: responseId)
    }
    
    /// Gets all active response IDs
    public var activeResponseIds: [UUID] {
        return Array(responses.keys.filter { !(responses[$0]?.isComplete ?? true) })
    }
    
    /// Gets statistics for all responses
    public var statistics: StreamCollectionStats {
        let allResponses = Array(responses.values)
        let completed = allResponses.filter { $0.isComplete }
        
        let totalTokens = allResponses.reduce(0) { $0 + $1.estimatedTokenCount }
        let averageLength = allResponses.isEmpty ? 0.0 : Double(totalTokens) / Double(allResponses.count)
        
        let completionReasons = completed.compactMap { $0.completionReason }
        let reasonCounts: [String: Int] = Dictionary(grouping: completionReasons, by: { $0.description })
            .mapValues { $0.count }
        
        return StreamCollectionStats(
            totalResponses: allResponses.count,
            completedResponses: completed.count,
            averageTokenLength: averageLength,
            completionReasons: reasonCounts
        )
    }
}

/// Statistics for stream collection
public struct StreamCollectionStats {
    public let totalResponses: Int
    public let completedResponses: Int
    public let averageTokenLength: Double
    public let completionReasons: [String: Int]
    
    public var completionRate: Double {
        return totalResponses > 0 ? Double(completedResponses) / Double(totalResponses) : 0.0
    }
}

// MARK: - Async Stream Extensions

extension AsyncThrowingStream where Element == StreamChunk {
    
    /// Collects all chunks into a complete response
    public func collectResponse() async throws -> StreamingResponse {
        var response = StreamingResponse()
        
        for try await chunk in self {
            response.addChunk(chunk)
            if chunk.isComplete {
                break
            }
        }
        
        return response
    }
    
    /// Collects only the content strings, ignoring metadata
    public func collectContent() async throws -> String {
        var content = ""
        
        for try await chunk in self {
            content += chunk.content
            if chunk.isComplete {
                break
            }
        }
        
        return content
    }
}

extension AsyncThrowingStream where Element == String {
    
    /// Converts a string stream to a chunk stream
    public func toChunkStream() -> AsyncThrowingStream<StreamChunk, Error> {
        return AsyncThrowingStream<StreamChunk, Error> { continuation in
            Task {
                do {
                    for try await content in self {
                        let chunk = StreamChunk.content(content)
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
