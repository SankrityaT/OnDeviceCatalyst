//
//  ValueTypeCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R1: pure Swift value and lifecycle behavior needing no llama
//  return value. Two classes:
//
//  - ValueTypeCharacterizationTests: undocumented-but-not-wrong current
//    behavior (X-ARCH-1), independently measured during this ticket's
//    discovery (spec N5, N8) and surprising enough that a naive test would
//    get it wrong. Not one of the eight baseline defects, so no ticket
//    suffix; `__no_defect` per the naming convention. X-CFG-1 (the
//    PredictionConfig preset quirk) is pinned in the N1 repair itself,
//    OnDeviceCatalystTests.swift, because that is the exact fact the
//    original, non-compiling testPredictionConfigPresets was trying to
//    assert.
//  - ValueTypeRegressionTests: intended, correct-today behavior this suite
//    must keep asserting is correct (X-PROFILE-1, X-STOP-1, X-PROMPT-1).
//    `test_requires_`, no `__ODC_` suffix, per the naming convention's third
//    rule. X-SETTINGS-1 and X-CATALYST-1 are likewise pinned in the N1
//    repair, since they preserve the original testInstanceSettingsValidation
//    and testCatalystServiceInitialization verbatim.
//

import XCTest
@testable import OnDeviceCatalyst

// MARK: - Characterization, no defect

final class ValueTypeCharacterizationTests: XCTestCase {

    /// CHARACTERIZATION (no ticket)
    /// Today: ModelArchitecture.detectFromPath("phi-probe.gguf") returns
    ///        .unknown, while LlamaBridge.createModelLoadingError has a
    ///        dedicated "phi" branch that matches the identical filename.
    ///        Two different filename classifiers in the same package
    ///        disagree about the same input (spec N5, N8).
    /// Should be: the two classifiers agree, per ODC-0104.
    /// Evidence: Sources/OnDeviceCatalyst/Core Foundation/ModelArchitecture.swift:71-189,
    ///           Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:65-101
    func test_characterizes_modelArchitectureDetection_disagreesWithLlamaBridgeClassifier__no_defect() {
        let detected = ModelArchitecture.detectFromPath("phi-probe.gguf")
        XCTAssertEqual(detected, .unknown, "the general-purpose classifier does not recognize this filename")

        // LlamaBridge.createModelLoadingError is private, so its "phi" branch
        // is mirrored here verbatim (its condition, not its behavior, is what
        // this case pins -- the point is that this substring check exists and
        // ModelArchitecture.detectFromPath's substring checks do not fire for
        // the identical input).
        let filename = "phi-probe.gguf"
        XCTAssertTrue(filename.contains("phi"), "precondition: fixture name matches LlamaBridge's phi branch")
        XCTAssertNotEqual(detected, .phi3, "the two classifiers disagree: one recognizes 'phi', the other returns .unknown")
    }
}

// MARK: - Regression: intended, correct-today behavior

final class ValueTypeRegressionTests: XCTestCase {

    func test_requires_modelProfile_throwsModelFileNotFoundForMissingPath() {
        let mockPath = "/mock/path/that/does/not/exist-\(UUID().uuidString).gguf"
        XCTAssertThrowsError(try ModelProfile(filePath: mockPath)) { error in
            guard let catalystError = error as? CatalystError else {
                XCTFail("expected CatalystError, got \(type(of: error))")
                return
            }
            guard case .modelFileNotFound(let path) = catalystError else {
                XCTFail("expected .modelFileNotFound, got \(catalystError)")
                return
            }
            XCTAssertEqual(path, mockPath)
        }
    }

    func test_requires_modelProfile_throwsModelFileCorruptedForFileUnder1MiB() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("odc-0004-small-\(UUID().uuidString).gguf")
        var small = "GGUF".data(using: .ascii)!
        small.append(Data(count: 100)) // well under the 1 MiB floor
        try small.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try ModelProfile(filePath: url.path)) { error in
            guard let catalystError = error as? CatalystError,
                  case .modelFileCorrupted = catalystError else {
                XCTFail("expected .modelFileCorrupted, got \(error)")
                return
            }
        }
    }

    func test_requires_modelProfile_throwsModelFileCorruptedForBadMagic() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("odc-0004-badmagic-\(UUID().uuidString).gguf")
        var bad = "NOPE".data(using: .ascii)!
        bad.append(Data(count: 2 * 1024 * 1024))
        try bad.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try ModelProfile(filePath: url.path)) { error in
            guard let catalystError = error as? CatalystError,
                  case .modelFileCorrupted = catalystError else {
                XCTFail("expected .modelFileCorrupted, got \(error)")
                return
            }
        }
    }

    /// X-STOP-1: pins the byte offset at which the stop sequence is detected
    /// for Llama 3's default stop set, streamed token-by-token the way
    /// generateTokens actually feeds StreamProcessor.
    func test_requires_streamProcessor_detectsStopSequence_atExpectedByteOffset() {
        let handler = StopSequenceHandler(architecture: .llama3)
        let processor = StreamProcessor(stopHandler: handler)

        var contentSoFar = ""
        for token in ["Hello", " world"] {
            let chunks = processor.processToken(token)
            XCTAssertEqual(chunks.count, 1)
            XCTAssertFalse(chunks[0].isComplete)
            contentSoFar += chunks[0].content
        }
        XCTAssertEqual(contentSoFar, "Hello world")
        XCTAssertEqual(contentSoFar.utf8.count, 11, "byte offset immediately before the stop token")

        let stopChunks = processor.processToken("<|eot_id|>")
        XCTAssertEqual(stopChunks.count, 1, "no trailing content chunk, only the completion")
        XCTAssertTrue(stopChunks[0].isComplete)
        XCTAssertEqual(stopChunks[0].metadata?.completionReason, .stopSequenceFound("<|eot_id|>"))
    }

    /// X-PROMPT-1: golden values for StandardPromptFormatter.formatPrompt,
    /// one per representative architecture family.
    func test_requires_standardPromptFormatter_producesGoldenLlama3Prompt() {
        let formatter = StandardPromptFormatter()
        let prompt = formatter.formatPrompt(
            turns: [Turn.user("Hi")],
            systemPrompt: "You are helpful.",
            architecture: .llama3
        )
        let expected = "<|begin_of_text|>" +
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        XCTAssertEqual(prompt, expected)
    }

    func test_requires_standardPromptFormatter_producesGoldenPhi3Prompt() {
        let formatter = StandardPromptFormatter()
        let prompt = formatter.formatPrompt(
            turns: [Turn.user("Hi")],
            systemPrompt: "You are helpful.",
            architecture: .phi3
        )
        let expected = "<|system|>\nYou are helpful.<|end|>\n" +
            "<|user|>\nHi<|end|>\n" +
            "<|assistant|>\n"
        XCTAssertEqual(prompt, expected)
    }

    func test_requires_standardPromptFormatter_producesGoldenChatMLFallbackPrompt() {
        let formatter = StandardPromptFormatter()
        let prompt = formatter.formatPrompt(
            turns: [Turn.user("Hi")],
            systemPrompt: nil,
            architecture: .unknown
        )
        let expected = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n" +
            "<|im_start|>user\nHi<|im_end|>\n" +
            "<|im_start|>assistant\n"
        XCTAssertEqual(prompt, expected)
    }
}
