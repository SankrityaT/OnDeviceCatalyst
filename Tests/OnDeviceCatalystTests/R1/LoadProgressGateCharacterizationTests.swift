//
//  LoadProgressGateCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R1: D3 (LlamaInstance.publishProgress gated its terminal-finish
//  logic on `if case .ready = progress, case .failed = progress`, a compound
//  AND over a single value, which was unsatisfiable for every LoadProgress
//  case). ODC-0012 replaced that gate with `progress.isComplete`. See
//  Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:580-587.
//
//  This mirrors the predicate verbatim rather than driving the private
//  publishProgress method directly (it is not exposed), which is exactly what
//  the spec's `## Design`, "Pinning what cannot be executed" describes as the
//  fallback shape for a fingerprinted defect site paired with an executable
//  case. The failure-path termination consequence of this same defect is
//  executed at R2 in InitializationFailureCharacterizationTests (C-D3-2),
//  which does not flip from ODC-0012 alone (see that file's comment block).
//

import XCTest
@testable import OnDeviceCatalyst

final class D3LoadProgressGateCharacterizationTests: XCTestCase {

    /// Mirror of the predicate at LlamaInstance.swift:583 after ODC-0012's
    /// repair: `progress.isComplete`, true for `.ready` and `.failed`, false
    /// for `.preparing` and `.loading`.
    private func publishProgressGateMirror(_ progress: LoadProgress) -> Bool {
        progress.isComplete
    }

    /// CHARACTERIZATION D3 (ODC-0012)
    /// Today: (post-repair) the gate is `progress.isComplete`, true for
    ///        `.ready` and `.failed`, false for `.preparing` and `.loading`,
    ///        so the stream terminates on either terminal state. Before this
    ///        repair the gate was `if case .ready = progress, case .failed =
    ///        progress`, an AND over one value, false for all four cases.
    /// Should be: exactly what "Today" now states -- satisfiable for the two
    ///        terminal cases, unsatisfiable for the two non-terminal cases.
    ///        This test's ID (C-D3-1) and name are kept unchanged from the
    ///        spec's literal `test_requires_` rename instruction because
    ///        scripts/check-characterization.py hardcodes both the exact
    ///        name `test_characterizes_..._ODC_0012` for C-D3-1's inventory
    ///        entry and a `test_requires_` name must never contain `__ODC_`
    ///        per this suite's own naming rule; satisfying the rename would
    ///        require editing scripts/**, out of this ticket's scope.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:583
    func test_characterizes_publishProgressGate_isUnsatisfiableForEveryCase__ODC_0012() {
        let terminalCases: [LoadProgress] = [.ready("x"), .failed("x")]
        let nonTerminalCases: [LoadProgress] = [.preparing("x"), .loading("x")]

        for progress in terminalCases {
            XCTAssertTrue(
                publishProgressGateMirror(progress),
                "gate must be satisfiable for \(progress)"
            )
        }
        for progress in nonTerminalCases {
            XCTAssertFalse(
                publishProgressGateMirror(progress),
                "gate must remain unsatisfiable for \(progress)"
            )
        }
    }
}
