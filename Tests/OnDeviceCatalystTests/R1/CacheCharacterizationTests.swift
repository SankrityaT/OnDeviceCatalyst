//
//  CacheCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R1: ModelCache read/write behavior. No llama return value is
//  consumed -- `LlamaInstance(profile:settings:predictionConfig:)` never
//  touches a backend, so `isReady` is `false` by construction. This is the
//  reader half of D1 (spec N6); the writer half (a genuinely ready instance
//  being cached and shut down by `Catalyst.releaseInstance`) needs real
//  inference and is R3 (`C-D1-3`).
//

import XCTest
@testable import OnDeviceCatalyst

final class D1CacheCharacterizationTests: XCTestCase {

    override func tearDown() {
        SyntheticModelFixture.removeAll()
        super.tearDown()
    }

    private func makeNotReadyInstance(suffix: String) -> (ModelProfile, InstanceSettings, LlamaInstance) {
        let profile = ModelProfile(
            mlxModelId: "odc-0004-fixture-\(suffix)-\(UUID().uuidString)",
            name: "ODC-0004 fixture"
        )
        let settings = InstanceSettings.balanced
        let instance = LlamaInstance(
            profile: profile,
            settings: settings,
            predictionConfig: .balanced
        )
        return (profile, settings, instance)
    }

    /// CHARACTERIZATION D1 (ODC-0010)
    /// Today: ModelCache.storeInstance/getInstance round-trip an instance
    ///        whose isReady is false; the cache performs no readiness check.
    /// Should be: getInstance returns nil, or the cache only ever holds ready
    ///        instances.
    /// Evidence: Sources/OnDeviceCatalyst/Service Layer/CacheSettings.swift:93-108
    func test_characterizes_modelCache_getInstance_returnsNotReadyInstance__ODC_0010() throws {
        let (profile, settings, instance) = makeNotReadyInstance(suffix: "d1-1")
        XCTAssertFalse(instance.isReady, "precondition: fixture instance must not be ready")

        ModelCache.shared.storeInstance(instance, for: profile, with: settings)

        // storeInstance writes under queue.async(flags: .barrier) (C-D1-2), so
        // poll with a bounded wait rather than assume ordering.
        let deadline = Date().addingTimeInterval(5.0)
        var retrieved: LlamaInstance?
        while Date() < deadline {
            if let found = ModelCache.shared.getInstance(for: profile, with: settings) {
                retrieved = found
                break
            }
            Thread.sleep(forTimeInterval: 0.05)
        }

        guard let cached = retrieved else {
            XCTFail("expected the store to become visible within the polling bound")
            return
        }
        XCTAssertFalse(cached.isReady, "the cache returned an instance whose isReady is false")

        ModelCache.shared.removeInstance(for: profile, with: settings)
    }

    /// CHARACTERIZATION D1 (ODC-0010)
    /// Today: ModelCache.storeInstance writes under queue.async(flags: .barrier),
    ///        so an immediate getInstance may legitimately return nil; a reader
    ///        must poll within a bounded wait rather than assume ordering.
    /// Should be: the store is ordered with respect to the read, or the API
    ///        documents and enforces the asynchrony explicitly.
    /// Evidence: Sources/OnDeviceCatalyst/Service Layer/CacheSettings.swift:111-139
    func test_characterizes_modelCache_storeInstance_isAsynchronous__ODC_0010() throws {
        let (profile, settings, instance) = makeNotReadyInstance(suffix: "d1-2")

        ModelCache.shared.storeInstance(instance, for: profile, with: settings)

        // Immediately after issuing the store, an immediate read is permitted
        // (not required) to see nil -- this is the observable asynchrony, and
        // this case pins that a bounded poll always eventually succeeds.
        let harnessPollInterval: TimeInterval = 0.05
        let harnessPollBound: TimeInterval = 5.0
        let deadline = Date().addingTimeInterval(harnessPollBound)
        var found = false
        while Date() < deadline {
            if ModelCache.shared.getInstance(for: profile, with: settings) != nil {
                found = true
                break
            }
            Thread.sleep(forTimeInterval: harnessPollInterval)
        }

        XCTAssertTrue(found, "the store must become visible to a polling reader within the bound")

        ModelCache.shared.removeInstance(for: profile, with: settings)
    }
}
