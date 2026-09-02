//
//  SyntheticModelFixture.swift
//  OnDeviceCatalystTests
//
//  ODC-0004: GGUF-magic fixture creation and teardown for R1/R2 cases.
//  Fixtures are written into NSTemporaryDirectory() at run time and removed by
//  removeAll(). Nothing is committed to the repository. See spec
//  "## Security and privacy".
//

import Foundation

public enum SyntheticModelFixture {

    private static var createdPaths: [String] = []
    private static let lock = NSLock()

    /// Creates a file of `bytes` length whose first four bytes are the GGUF
    /// magic ("GGUF"), so it passes both `ModelProfile.validateModel()` and
    /// `LlamaBridge.validateModelFile`. The remainder is zero-filled.
    ///
    /// - Parameter named: the filename to use. Filename content is significant:
    ///   `LlamaBridge.createModelLoadingError` branches on it (spec N5), so
    ///   callers choosing "phi-*.gguf" vs "generic-*.gguf" select which error
    ///   class a load failure produces.
    @discardableResult
    public static func make(named: String, bytes: Int = 2 * 1024 * 1024) throws -> URL {
        precondition(bytes >= 4, "fixture must be at least 4 bytes to carry the GGUF magic")
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("odc-0004-\(UUID().uuidString)-\(named)")

        var data = "GGUF".data(using: .ascii)!
        data.append(Data(count: bytes - data.count))

        try data.write(to: url, options: .atomic)

        lock.lock()
        createdPaths.append(url.path)
        lock.unlock()

        return url
    }

    /// Removes every fixture created by `make(named:bytes:)` in this process.
    public static func removeAll() {
        lock.lock()
        let paths = createdPaths
        createdPaths.removeAll()
        lock.unlock()

        for path in paths {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
}
