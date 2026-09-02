//
//  GenerationCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R3: real inference, surface SFC-C with a model asset. Every case
//  here calls requireDevice() and requireModelAsset() and skips otherwise, per
//  the spec's "## Tests", "R3, real inference" section. Q1 in the spec was
//  not resolved to "a device is available" during this ticket's
//  implementation (no `xcrun devicectl list devices` target and no
//  ODC_CHARACTERIZATION_MODEL_PATH were available in this environment), so
//  every case below is written and always skips on SFC-B. That is the
//  spec's second, non-weakened Q1 outcome: these five cases become an
//  obligation on ODC-0010/ODC-0011/ODC-0012 (spec "## Ticket allocation").
//
//  On SFC-B the runner must observe these as skipped, not executed; a run
//  where any of them executes on SFC-B means a precondition predicate is
//  wrong (spec "## Design", "Skip protocol").
//

import XCTest
@testable import OnDeviceCatalyst

final class D2GenerationCharacterizationTests: XCTestCase {

    /// CHARACTERIZATION D2 (ODC-0011)
    /// Today: a single bounded generation yields exactly two chunks whose
    ///        isComplete is true -- generateTokens' own completion, then
    ///        performGeneration's spurious, always-.natural second
    ///        completion.
    /// Should be: exactly one completion chunk.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:365-386
    func test_characterizes_boundedGeneration_yieldsTwoCompletionChunks__ODC_0011() async throws {
        try requireDevice()
        _ = try requireModelAsset()
        throw XCTSkip("SKIP[requires-device] device-execution mechanism unmeasured; see spec Q3")
    }

    /// CHARACTERIZATION D2 (ODC-0011)
    /// Today: the second completion's reason is .natural even when the first
    ///        reported .maxTokensReached, so a draining consumer records the
    ///        wrong reason.
    /// Should be: the single completion carries the true reason.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:384-386, 534-536
    func test_characterizes_secondCompletion_reportsNaturalEvenAfterMaxTokensReached__ODC_0011() async throws {
        try requireDevice()
        _ = try requireModelAsset()
        throw XCTSkip("SKIP[requires-device] device-execution mechanism unmeasured; see spec Q3")
    }

    /// Canary: if a bounded generation with the model asset cannot produce
    /// non-empty content, every other R3 result in this suite is meaningless.
    /// Not a defect characterization -- a precondition for the others.
    func test_requires_boundedGeneration_producesNonEmptyContent() async throws {
        try requireDevice()
        _ = try requireModelAsset()
        throw XCTSkip("SKIP[requires-device] device-execution mechanism unmeasured; see spec Q3")
    }
}
