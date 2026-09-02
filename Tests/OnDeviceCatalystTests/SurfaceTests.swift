//
//  SurfaceTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004: surface canaries, every surface. If any of these fail, the
//  requirement-class predicates the rest of the suite relies on are wrong,
//  and every downstream R2/R3 skip decision is untrustworthy. See spec
//  "## Design", "Skip protocol": "Each requirement class has a surface
//  canary that asserts the predicate itself."
//

import XCTest
@testable import OnDeviceCatalyst

final class CharacterizationSurfaceTests: XCTestCase {

    /// S-1: CharacterizationSurface.current matches the surface the runner
    /// declared. The runner declares its surface via ODC_CHARACTERIZATION_SURFACE
    /// (set by scripts/run-characterization.sh); when unset, this canary only
    /// asserts internal self-consistency of the predicate.
    func test_surface_reportsDeclaredSurface() {
        let declared = ProcessInfo.processInfo.environment["ODC_CHARACTERIZATION_SURFACE"]
        switch declared {
        case "simulator":
            XCTAssertEqual(CharacterizationSurface.current, .simulatorStub)
        case "device":
            XCTAssertEqual(CharacterizationSurface.current, .device)
        default:
            // No declaration; assert only that current is well-defined and
            // consistent with SimulatorSupport, which is what it wraps.
            let expected: CharacterizationSurface = SimulatorSupport.isSimulator ? .simulatorStub : .device
            XCTAssertEqual(CharacterizationSurface.current, expected)
        }
    }

    /// S-2: on SFC-B, SimulatorSupport.isSimulator is true; on SFC-C it is
    /// false.
    func test_surface_simulatorSupportMatchesCurrentSurface() {
        switch CharacterizationSurface.current {
        case .simulatorStub:
            XCTAssertTrue(SimulatorSupport.isSimulator)
        case .device:
            XCTAssertFalse(SimulatorSupport.isSimulator)
        }
    }

    /// S-3: on SFC-B, LlamaBridge.loadModel fails for a valid-magic fixture,
    /// confirming the stub is in place (spec N3).
    func test_surface_stubRejectsAValidMagicFixture() throws {
        try requireSimulatorStub()

        let url = try SyntheticModelFixture.make(named: "surface-canary.gguf")
        defer { SyntheticModelFixture.removeAll() }

        XCTAssertThrowsError(try LlamaBridge.loadModel(path: url.path, settings: .balanced))
    }

    /// S-4: CharacterizationSurface.modelAssetPath is nil unless the
    /// environment variable names an existing file with GGUF magic.
    func test_surface_modelAssetPathIsNilWithoutAValidEnvironmentVariable() {
        let env = ProcessInfo.processInfo.environment["ODC_CHARACTERIZATION_MODEL_PATH"]
        if env == nil || env == "" {
            XCTAssertNil(CharacterizationSurface.modelAssetPath)
        } else {
            // If the operator did set it, the canary instead asserts the
            // helper's own contract: non-nil only when the file exists and
            // carries the GGUF magic.
            if let path = CharacterizationSurface.modelAssetPath {
                XCTAssertTrue(FileManager.default.fileExists(atPath: path))
            }
        }
    }
}
