//
//  LoadProgressGateCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R1: D3 (LlamaInstance.publishProgress gates its terminal-finish
//  logic on `if case .ready = progress, case .failed = progress`, a compound
//  AND over a single value, which is unsatisfiable for every LoadProgress
//  case). See Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:580-587.
//
//  This mirrors the predicate verbatim rather than driving the private
//  publishProgress method directly (it is not exposed), which is exactly what
//  the spec's `## Design`, "Pinning what cannot be executed" describes as the
//  fallback shape for a fingerprinted defect site paired with an executable
//  case. The failure-path termination consequence of this same defect is
//  executed at R2 in InitializationFailureCharacterizationTests (C-D3-2).
//

import XCTest
@testable import OnDeviceCatalyst

final class D3LoadProgressGateCharacterizationTests: XCTestCase {

    /// Verbatim transcription of the predicate at LlamaInstance.swift:583:
    /// `if case .ready = progress, case .failed = progress`. A comma between
    /// pattern-match conditions in a single `if` is a logical AND, and no
    /// `LoadProgress` value can be both `.ready` and `.failed` simultaneously,
    /// so this predicate is `false` for every possible input.
    private func publishProgressGateMirror(_ progress: LoadProgress) -> Bool {
        if case .ready = progress, case .failed = progress {
            return true
        }
        return false
    }

    /// CHARACTERIZATION D3 (ODC-0012)
    /// Today: the gate `if case .ready = progress, case .failed = progress` is
    ///        an AND over one value and is false for all four LoadProgress
    ///        cases, so publishProgress never finishes loadingContinuation via
    ///        this branch.
    /// Should be: the predicate is true for .ready and for .failed (an OR, or
    ///        two separate branches), so the stream terminates on either
    ///        terminal state.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:583
    func test_characterizes_publishProgressGate_isUnsatisfiableForEveryCase__ODC_0012() {
        let cases: [LoadProgress] = [
            .preparing("x"),
            .loading("x"),
            .ready("x"),
            .failed("x"),
        ]
        for progress in cases {
            XCTAssertFalse(
                publishProgressGateMirror(progress),
                "gate must be unsatisfiable for \(progress)"
            )
        }
    }
}
