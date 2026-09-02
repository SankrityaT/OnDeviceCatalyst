//
//  CharacterizationSurface.swift
//  OnDeviceCatalystTests
//
//  ODC-0004: surface detection and skip helpers for the characterization suite.
//  See docs/specs/ODC-0004-v2-characterization-suite.md, "## Interfaces / Support API".
//

import Foundation
import Metal
import XCTest
@testable import OnDeviceCatalyst

/// Identifies the execution surface a characterization case is running on.
///
/// This does not attempt to distinguish a "real" llama.cpp slice from the
/// simulator stub by inspecting binary content; it uses the same predicate the
/// runtime itself uses (`SimulatorSupport.isSimulator`), because that predicate
/// is what determines which slice SwiftPM/Xcode actually linked. See spec N2/N3.
public enum CharacterizationSurface: Equatable {
    case simulatorStub
    case device

    /// The surface this process is actually running on.
    public static var current: CharacterizationSurface {
        SimulatorSupport.isSimulator ? .simulatorStub : .device
    }

    /// True when the linked llama.cpp slice is the 51-symbol simulator stub
    /// described in the spec's N2/N3 (returns null/zero, never traps).
    /// False on a physical device, where the real slice is linked.
    public static var isSimulatorStub: Bool {
        current == .simulatorStub
    }

    /// True when the process is linked against a real llama.cpp slice capable of
    /// inference, i.e. running on a physical device (`SFC-C`).
    public static var hasRealLlamaSlice: Bool {
        current == .device
    }

    /// Reads `ODC_CHARACTERIZATION_MODEL_PATH`. Returns the path only if it
    /// names an existing file whose first four bytes are the GGUF magic.
    /// Never searches the filesystem and never embeds a default path -- that is
    /// the mistake this ticket pins in `Tests/BERTEmbeddingTest.swift` (N9).
    public static var modelAssetPath: String? {
        guard let path = ProcessInfo.processInfo.environment["ODC_CHARACTERIZATION_MODEL_PATH"],
              !path.isEmpty else {
            return nil
        }
        guard FileManager.default.fileExists(atPath: path) else {
            return nil
        }
        guard let handle = FileHandle(forReadingAtPath: path) else {
            return nil
        }
        defer { handle.closeFile() }
        let magic = handle.readData(ofLength: 4)
        guard magic == "GGUF".data(using: .ascii)! else {
            return nil
        }
        return path
    }

    /// True when a Metal device is available in this process.
    public static var hasMetalDevice: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }
}

extension XCTestCase {

    /// Skips unless a physical device is the current surface.
    /// `SKIP[requires-device]` per the spec's skip protocol.
    public func requireDevice(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            CharacterizationSurface.hasRealLlamaSlice,
            "SKIP[requires-device] this case needs a physical device with the real llama.cpp slice",
            file: file,
            line: line
        )
    }

    /// Skips unless `ODC_CHARACTERIZATION_MODEL_PATH` names a real GGUF asset.
    /// `SKIP[requires-model-asset]` per the spec's skip protocol.
    @discardableResult
    public func requireModelAsset(file: StaticString = #filePath, line: UInt = #line) throws -> String {
        guard let path = CharacterizationSurface.modelAssetPath else {
            throw XCTSkip(
                "SKIP[requires-model-asset] set ODC_CHARACTERIZATION_MODEL_PATH to an existing GGUF file",
                file: file,
                line: line
            )
        }
        return path
    }

    /// Skips unless the current surface is the simulator stub slice.
    /// `SKIP[requires-simulator-stub]` per the spec's skip protocol.
    public func requireSimulatorStub(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            CharacterizationSurface.isSimulatorStub,
            "SKIP[requires-simulator-stub] this case characterizes the stub's deterministic null return",
            file: file,
            line: line
        )
    }

    /// Skips unless a Metal-capable device is available.
    /// `SKIP[requires-metal-device]` per the spec's skip protocol.
    public func requireMetalDevice(file: StaticString = #filePath, line: UInt = #line) throws {
        try XCTSkipUnless(
            CharacterizationSurface.hasMetalDevice,
            "SKIP[requires-metal-device] no Metal device is available in this process",
            file: file,
            line: line
        )
    }
}
