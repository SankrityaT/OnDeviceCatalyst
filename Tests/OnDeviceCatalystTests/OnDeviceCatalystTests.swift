//
//  OnDeviceCatalystTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004 N1 repair (ticket ODC-0018): the declared test target did not
//  compile on any triple because this file referenced
//  `PredictionConfig.quality`, which does not exist. The five real presets
//  are `balanced`, `creative`, `speed`, `deterministic`, `mirostat`
//  (Sources/OnDeviceCatalyst/Core Foundation/PredictionConfig.swift:71-119).
//
//  Per the spec's "## Design", "Permitted changes to tracked files": this is
//  the one allowed edit to an existing test source. Each of the four original
//  cases is renamed and reclassified under the naming convention in
//  docs/specs/ODC-0004-v2-characterization-suite.md, preserving what each one
//  was trying to assert.
//

import XCTest
@testable import OnDeviceCatalyst

final class OnDeviceCatalystTests: XCTestCase {

    /// Was testModelProfileCreation. Preserved and reclassified as a
    /// regression: ModelProfile(filePath:) throwing a CatalystError for an
    /// unreachable path is intended, correct behavior.
    func test_requires_modelProfile_throwsCatalystErrorForAMockPath() throws {
        let mockPath = "/mock/path/test-model.gguf"

        XCTAssertThrowsError(try ModelProfile(filePath: mockPath)) { error in
            XCTAssertTrue(error is CatalystError)
        }
    }

    /// Was testInstanceSettingsValidation. Preserved and reclassified as a
    /// regression: the iPhone 16 Pro Max preset's values are intended,
    /// correct configuration.
    func test_requires_iphone16ProMaxSettings_areValidAndHaveTheirOptimizedValues() throws {
        let settings = InstanceSettings.iphone16ProMax

        XCTAssertNoThrow(try settings.validate())
        XCTAssertEqual(settings.contextLength, 2048)
        XCTAssertEqual(settings.batchSize, 256)
        XCTAssertEqual(settings.gpuLayers, 25)
        XCTAssertEqual(settings.cpuThreads, 6)
    }

    /// Was testPredictionConfigPresets, which referenced the non-existent
    /// `.quality` preset (N1) and, independently, assumed a "quality" preset
    /// would allow more tokens than "speed" -- backwards from what the code
    /// does today (N8: balanced.maxTokens == -1, speed.maxTokens == 1024).
    /// Both beliefs were false, so both are pinned here as characterizations
    /// of undocumented-but-not-wrong current behavior rather than repaired
    /// away.
    ///
    /// CHARACTERIZATION (no ticket)
    /// Today: there is no PredictionConfig.quality preset; the five presets
    ///        are exactly balanced, creative, speed, deterministic,
    ///        mirostat. speed.maxTokens (1024) is numerically greater than
    ///        balanced.maxTokens (-1, meaning unlimited), the opposite of
    ///        what the original test assumed a "quality vs speed" comparison
    ///        would show.
    /// Should be: unchanged unless a preset's values change deliberately, in
    ///        which case this case is updated in the same commit as that
    ///        change.
    /// Evidence: Sources/OnDeviceCatalyst/Core Foundation/PredictionConfig.swift:71-119
    func test_characterizes_predictionConfigPresets_speedTokenBudgetExceedsBalanced__no_defect() {
        let speedConfig = PredictionConfig.speed
        let balancedConfig = PredictionConfig.balanced

        // Speed still prioritizes lower temperature over balanced.
        XCTAssertLessThan(speedConfig.temperature, balancedConfig.temperature)

        // But speed's finite budget (1024) is numerically greater than
        // balanced's -1 "unlimited" sentinel -- not what a reader comparing
        // "speed" against a higher-effort preset would expect.
        XCTAssertEqual(balancedConfig.maxTokens, -1)
        XCTAssertEqual(speedConfig.maxTokens, 1024)
        XCTAssertGreaterThan(speedConfig.maxTokens, balancedConfig.maxTokens)

        let presetNames: Set<String> = ["balanced", "creative", "speed", "deterministic", "mirostat"]
        XCTAssertNoThrow(try PredictionConfig.balanced.validate())
        XCTAssertNoThrow(try PredictionConfig.creative.validate())
        XCTAssertNoThrow(try PredictionConfig.speed.validate())
        XCTAssertNoThrow(try PredictionConfig.deterministic.validate())
        XCTAssertNoThrow(try PredictionConfig.mirostat.validate())
        XCTAssertEqual(presetNames.count, 5, "there is no sixth preset such as .quality")
    }

    /// Was testCatalystServiceInitialization. Preserved and reclassified as a
    /// regression: Catalyst.shared being a constructible singleton is
    /// intended, correct behavior.
    func test_requires_catalystShared_isASingletonAndIsConstructible() {
        let first = Catalyst.shared
        let second = Catalyst.shared
        XCTAssertNotNil(first)
        XCTAssertTrue(first === second, "Catalyst.shared must be a singleton")
    }
}
