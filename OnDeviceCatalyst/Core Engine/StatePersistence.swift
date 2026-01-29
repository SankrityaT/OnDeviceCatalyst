//
//  StatePersistence.swift
//  OnDeviceCatalyst
//
//  Utilities for saving and loading conversation state for efficient context reuse
//

import Foundation

/// Manages saving and loading of model context state
/// Allows resuming conversations without re-processing the entire context
public class StatePersistence {

    // MARK: - State Snapshot

    /// A snapshot of the model's state that can be saved and restored
    public struct StateSnapshot {
        /// The raw state data
        public let data: Data
        /// Number of tokens that were processed when this snapshot was taken
        public let tokenCount: Int
        /// Timestamp when the snapshot was created
        public let timestamp: Date
        /// Model identifier this snapshot belongs to
        public let modelId: String

        /// Size of the state data in bytes
        public var sizeInBytes: Int {
            return data.count
        }

        /// Size of the state data formatted as string
        public var formattedSize: String {
            let formatter = ByteCountFormatter()
            formatter.allowedUnits = [.useKB, .useMB]
            formatter.countStyle = .file
            return formatter.string(fromByteCount: Int64(data.count))
        }
    }

    // MARK: - Save State

    /// Capture current state from a context
    /// - Parameters:
    ///   - context: The llama context to capture state from
    ///   - tokenCount: Number of tokens currently in context
    ///   - modelId: Identifier for the model
    /// - Returns: A state snapshot that can be persisted
    public static func captureState(
        from context: CContext,
        tokenCount: Int,
        modelId: String
    ) throws -> StateSnapshot {
        let stateSize = LlamaBridge.getStateSize(context)

        guard stateSize > 0 else {
            throw CatalystError.cacheOperationFailed(
                operation: "captureState",
                reason: "State size is zero"
            )
        }

        // Allocate buffer
        var buffer = [UInt8](repeating: 0, count: stateSize)

        // Copy state to buffer
        let writtenBytes = buffer.withUnsafeMutableBufferPointer { ptr -> Int in
            guard let baseAddress = ptr.baseAddress else { return 0 }
            return LlamaBridge.saveState(context, to: baseAddress, size: stateSize)
        }

        guard writtenBytes > 0 else {
            throw CatalystError.cacheOperationFailed(
                operation: "captureState",
                reason: "Failed to write state data"
            )
        }

        let data = Data(buffer.prefix(writtenBytes))

        return StateSnapshot(
            data: data,
            tokenCount: tokenCount,
            timestamp: Date(),
            modelId: modelId
        )
    }

    /// Save state snapshot to a file
    /// - Parameters:
    ///   - snapshot: The state snapshot to save
    ///   - url: File URL to save to
    public static func saveToFile(_ snapshot: StateSnapshot, at url: URL) throws {
        // Create wrapper for serialization
        let wrapper = StateFileWrapper(
            stateData: snapshot.data,
            tokenCount: snapshot.tokenCount,
            timestamp: snapshot.timestamp,
            modelId: snapshot.modelId
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let jsonData = try encoder.encode(wrapper)

        try jsonData.write(to: url, options: .atomic)

        print("StatePersistence: Saved state (\(snapshot.formattedSize)) to \(url.lastPathComponent)")
    }

    // MARK: - Load State

    /// Load state snapshot from a file
    /// - Parameter url: File URL to load from
    /// - Returns: The loaded state snapshot
    public static func loadFromFile(at url: URL) throws -> StateSnapshot {
        let jsonData = try Data(contentsOf: url)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let wrapper = try decoder.decode(StateFileWrapper.self, from: jsonData)

        return StateSnapshot(
            data: wrapper.stateData,
            tokenCount: wrapper.tokenCount,
            timestamp: wrapper.timestamp,
            modelId: wrapper.modelId
        )
    }

    /// Restore state to a context
    /// - Parameters:
    ///   - snapshot: The state snapshot to restore
    ///   - context: The context to restore state to
    /// - Returns: Number of bytes read
    public static func restoreState(
        _ snapshot: StateSnapshot,
        to context: CContext
    ) throws -> Int {
        let readBytes = snapshot.data.withUnsafeBytes { ptr -> Int in
            guard let baseAddress = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return 0
            }
            return LlamaBridge.loadState(context, from: baseAddress, size: snapshot.data.count)
        }

        guard readBytes > 0 else {
            throw CatalystError.cacheOperationFailed(
                operation: "restoreState",
                reason: "Failed to read state data"
            )
        }

        print("StatePersistence: Restored state (\(snapshot.formattedSize), \(snapshot.tokenCount) tokens)")

        return readBytes
    }

    // MARK: - Validation

    /// Check if a snapshot is compatible with current context
    /// - Parameters:
    ///   - snapshot: The snapshot to validate
    ///   - modelId: Current model identifier
    ///   - maxAge: Maximum age of snapshot in seconds (default 1 hour)
    /// - Returns: True if snapshot can be used
    public static func isSnapshotValid(
        _ snapshot: StateSnapshot,
        forModelId modelId: String,
        maxAge: TimeInterval = 3600
    ) -> Bool {
        // Check model ID matches
        guard snapshot.modelId == modelId else {
            print("StatePersistence: Snapshot model ID mismatch")
            return false
        }

        // Check age
        let age = Date().timeIntervalSince(snapshot.timestamp)
        guard age < maxAge else {
            print("StatePersistence: Snapshot too old (\(Int(age))s)")
            return false
        }

        // Check data is not empty
        guard !snapshot.data.isEmpty else {
            print("StatePersistence: Snapshot data is empty")
            return false
        }

        return true
    }

    // MARK: - File Management

    /// Get the default directory for state files
    public static var defaultStateDirectory: URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("OnDeviceCatalyst/States", isDirectory: true)
    }

    /// Ensure the state directory exists
    public static func ensureDirectoryExists() throws {
        try FileManager.default.createDirectory(
            at: defaultStateDirectory,
            withIntermediateDirectories: true,
            attributes: nil
        )
    }

    /// Generate a filename for a state snapshot
    public static func generateFilename(modelId: String, conversationId: String? = nil) -> String {
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let convPart = conversationId.map { "_\($0)" } ?? ""
        return "\(modelId)\(convPart)_\(timestamp).state"
    }

    /// Clean up old state files
    /// - Parameter maxAge: Maximum age in seconds (default 24 hours)
    public static func cleanupOldStates(maxAge: TimeInterval = 86400) {
        do {
            let fileManager = FileManager.default
            let stateDir = defaultStateDirectory

            guard fileManager.fileExists(atPath: stateDir.path) else { return }

            let files = try fileManager.contentsOfDirectory(
                at: stateDir,
                includingPropertiesForKeys: [.creationDateKey],
                options: .skipsHiddenFiles
            )

            let now = Date()
            var removedCount = 0

            for file in files where file.pathExtension == "state" {
                if let attrs = try? fileManager.attributesOfItem(atPath: file.path),
                   let creationDate = attrs[.creationDate] as? Date,
                   now.timeIntervalSince(creationDate) > maxAge {
                    try? fileManager.removeItem(at: file)
                    removedCount += 1
                }
            }

            if removedCount > 0 {
                print("StatePersistence: Cleaned up \(removedCount) old state files")
            }
        } catch {
            print("StatePersistence: Cleanup error: \(error)")
        }
    }
}

// MARK: - File Wrapper

/// Internal wrapper for JSON serialization
private struct StateFileWrapper: Codable {
    let stateData: Data
    let tokenCount: Int
    let timestamp: Date
    let modelId: String
}
