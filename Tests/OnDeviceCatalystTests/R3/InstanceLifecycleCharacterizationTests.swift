//
//  InstanceLifecycleCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R3: real inference, surface SFC-C with a model asset. See the
//  header of GenerationCharacterizationTests.swift for why every case here
//  always skips on SFC-B under this revision's Q1 disposition
//  (specified-unexecuted).
//

import XCTest
@testable import OnDeviceCatalyst

final class D3ReadyStreamLifecycleCharacterizationTests: XCTestCase {

    /// CHARACTERIZATION D3 (ODC-0012)
    /// Today: after .ready is delivered, the loading stream does not
    ///        terminate within a bounded wait, so a consumer awaiting
    ///        termination hangs -- the unsatisfiable gate at publishProgress
    ///        (:583) never finishes loadingContinuation on the success path
    ///        either.
    /// Should be: the stream finishes after .ready.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:580-587
    func test_characterizes_afterReady_loadingStream_doesNotTerminate__ODC_0012() async throws {
        try requireDevice()
        _ = try requireModelAsset()
        throw XCTSkip("SKIP[requires-device] device-execution mechanism unmeasured; see spec Q3")
    }
}

final class D1ReleaseInstanceCharacterizationTests: XCTestCase {

    /// CHARACTERIZATION D1 (ODC-0010)
    /// Today: after releaseInstance drops the last reference, the instance is
    ///        inserted into the cache and then shut down (Catalyst.swift:507,
    ///        510-512), so a subsequent cache read returns an instance that
    ///        is not ready. This is the writer half of D1; the reader half is
    ///        executable at R1 with no ready instance at all (C-D1-1, C-D1-2).
    /// Should be: the cached instance stays ready, or is not cached.
    /// Evidence: Sources/OnDeviceCatalyst/Service Layer/Catalyst.swift:494-522
    func test_characterizes_releaseInstance_cachesThenShutsDownAReadyInstance__ODC_0010() async throws {
        try requireDevice()
        _ = try requireModelAsset()
        throw XCTSkip("SKIP[requires-device] device-execution mechanism unmeasured; see spec Q3")
    }
}
