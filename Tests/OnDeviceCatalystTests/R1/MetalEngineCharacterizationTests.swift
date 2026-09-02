//
//  MetalEngineCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R1: D5 consequence half. SwiftPM declares zero resources on the
//  OnDeviceCatalyst target and never compiles the seven .metal shader files
//  (packaging half is C-D5-1/2/3, checker-only, SFC-A). The consequence is
//  that MetalComputeEngine's init throws, because device.makeDefaultLibrary()
//  finds no compiled shader library. See
//  Sources/OnDeviceCatalyst/Metal Engine/Compute/MetalComputeEngine.swift:77-135.
//
//  Q2 (open at spec approval time) leaves it undecided whether an Xcode-built
//  consumer would also fail, and if so with which of two messages (absent
//  Metal device vs. absent shader library) -- see spec "## Open questions",
//  Q2. This case therefore asserts only that construction throws, not which
//  message is produced.
//

import XCTest
@testable import OnDeviceCatalyst

final class D5MetalEngineCharacterizationTests: XCTestCase {

    /// CHARACTERIZATION D5 (ODC-0014)
    /// Today: MetalComputeEngine() throws CatalystError.unknown, because
    ///        SwiftPM does not compile or bundle the .metal shaders (no
    ///        default.metallib exists in the build product), so
    ///        device.makeDefaultLibrary() returns nil. Which of the two
    ///        possible messages (absent device vs. absent library) is
    ///        produced is Q2, open per the spec, so only the throw itself and
    ///        its error type are pinned here.
    /// Should be: construction succeeds on a Metal-capable device once the
    ///        shaders are declared as SwiftPM resources and the loader reads
    ///        Bundle.module.
    /// Evidence: Sources/OnDeviceCatalyst/Metal Engine/Compute/MetalComputeEngine.swift:88-92
    func test_characterizes_metalComputeEngineInit_throwsBecauseShaderLibraryIsUnpackaged__ODC_0014() {
        XCTAssertThrowsError(try MetalComputeEngine()) { error in
            guard let catalystError = error as? CatalystError else {
                XCTFail("expected a CatalystError, got \(type(of: error))")
                return
            }
            guard case .unknown(let details) = catalystError else {
                XCTFail("expected CatalystError.unknown, got \(catalystError)")
                return
            }
            XCTAssertFalse(details.isEmpty, "the unknown-error message should not be empty")
        }
    }
}
