//
//  InitializationFailureCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R2: a llama_* call's failure result is consumed. On SFC-B this is
//  satisfied by the simulator stub (spec N3: `_llama_load_model_from_file`
//  returns null unconditionally, with no trap), which makes
//  `LlamaBridge.loadModel` deterministically fail for any file that passes
//  preflight validation. That determinism is what lets D8's consumer-visible
//  symptom and D3's failure-path termination be executed without a model or a
//  device (spec N5).
//
//  Liveness bound: `loadingStreamTimeout` below is generous and is a harness
//  parameter, not a performance claim (spec "## Benchmarks"). A recording that
//  does not terminate within it fails the case with `harness-defect`
//  semantics (the stream should always finish, just never with a terminal
//  event -- that absence is the thing being pinned).
//

import XCTest
@testable import OnDeviceCatalyst

final class D8InitializationFailureCharacterizationTests: XCTestCase {

    /// Liveness bound, not a performance claim. See spec "## Benchmarks".
    private static let loadingStreamTimeout: Duration = .seconds(20)

    override func tearDown() {
        SyntheticModelFixture.removeAll()
        super.tearDown()
    }

    private func recordLoadingStream(fixtureName: String) async throws -> StreamRecorder<LoadProgress>.Recording {
        let url = try SyntheticModelFixture.make(named: fixtureName)
        let profile = try ModelProfile(filePath: url.path)
        let instance = LlamaInstance(profile: profile, settings: .balanced, predictionConfig: .balanced)
        let stream = instance.initialize()
        let recorder = StreamRecorder<LoadProgress>()
        return await recorder.record(stream, timeout: Self.loadingStreamTimeout)
    }

    private func assertNoTerminalEvent(_ recording: StreamRecorder<LoadProgress>.Recording, file: StaticString = #filePath, line: UInt = #line) {
        let sawTerminal = recording.events.contains { progress in
            switch progress {
            case .ready, .failed: return true
            case .preparing, .loading: return false
            }
        }
        XCTAssertFalse(sawTerminal, "neither .ready nor .failed should ever be delivered", file: file, line: line)
    }

    /// CHARACTERIZATION D8 (ODC-0015)
    /// Today: with a recoverable-class fixture (filename contains "phi", so
    ///        LlamaBridge.createModelLoadingError classifies it
    ///        .architectureUnsupported, isRecoverable == true), the loading
    ///        stream yields exactly preparing, loading, loading, then
    ///        terminates. Neither .ready nor .failed is ever delivered,
    ///        because handleInitializationError's cleanup() (:188) finishes
    ///        loadingContinuation before attemptFallbackInitialization's own
    ///        publishProgress calls run against it.
    /// Should be: the fallback path's own progress events are delivered,
    ///        ending in .ready or .failed.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:184-229
    func test_characterizes_recoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015() async throws {
        try requireSimulatorStub()
        let recording = try await recordLoadingStream(fixtureName: "phi-probe.gguf")

        XCTAssertTrue(recording.terminated, "the stream must finish (via cleanup()) within the liveness bound")
        assertNoTerminalEvent(recording)

        let messages = recording.events.map { $0.message }
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages.first, "Validating model file")
        XCTAssertEqual(messages.dropFirst().first, "Initializing llama.cpp backend")
        XCTAssertTrue(messages.last?.hasPrefix("Loading model from") ?? false)
    }

    /// CHARACTERIZATION D8 (ODC-0015)
    /// Today: with a non-recoverable-class fixture (generic filename ->
    ///        .modelLoadingFailed, isRecoverable == false), the sequence is
    ///        identical to the recoverable case above: preparing, loading,
    ///        loading, then termination with no terminal event. The
    ///        recoverable and non-recoverable classes are externally
    ///        indistinguishable to any consumer of the stream.
    /// Should be: the non-recoverable path delivers .failed directly, and the
    ///        two classes become distinguishable.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:184-196
    func test_characterizes_nonRecoverableFailure_loadingStream_deliversNoTerminalEvent__ODC_0015() async throws {
        try requireSimulatorStub()
        let recording = try await recordLoadingStream(fixtureName: "generic-probe.gguf")

        XCTAssertTrue(recording.terminated, "the stream must finish (via cleanup()) within the liveness bound")
        assertNoTerminalEvent(recording)

        let messages = recording.events.map { $0.message }
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages.first, "Validating model file")
        XCTAssertEqual(messages.dropFirst().first, "Initializing llama.cpp backend")
        XCTAssertTrue(messages.last?.hasPrefix("Loading model from") ?? false)
    }

    /// CHARACTERIZATION D8 (ODC-0015)
    /// Today: the recoverable-class fixture's event sequence and the
    ///        non-recoverable-class fixture's event sequence are identical in
    ///        shape, so a consumer cannot tell which internal branch was
    ///        taken from the stream alone. Recorded honestly: this observes
    ///        the event sequences, not which internal branch fired -- that
    ///        the two are indistinguishable is precisely the finding (spec N5).
    /// Should be: the two classes are distinguishable, and the non-recoverable
    ///        path delivers .failed.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:184-229
    func test_characterizes_recoverableAndNonRecoverableFailures_areExternallyIndistinguishable__ODC_0015() async throws {
        try requireSimulatorStub()
        let recoverable = try await recordLoadingStream(fixtureName: "phi-probe.gguf")
        let nonRecoverable = try await recordLoadingStream(fixtureName: "generic-probe.gguf")

        let recoverableShape = recoverable.events.map { messageKind($0) }
        let nonRecoverableShape = nonRecoverable.events.map { messageKind($0) }

        XCTAssertEqual(
            recoverableShape,
            nonRecoverableShape,
            "the two error classes produce the same observable sequence shape"
        )
    }

    private func messageKind(_ progress: LoadProgress) -> String {
        switch progress {
        case .preparing: return "preparing"
        case .loading: return "loading"
        case .ready: return "ready"
        case .failed: return "failed"
        }
    }

    /// CHARACTERIZATION D8 (ODC-0015)
    /// Today: after the loading stream ends, isReady is false and no further
    ///        event arrives -- the instance is left in a not-ready state with
    ///        no diagnostic ever surfaced to a caller consuming the stream.
    /// Should be: a terminal event precedes termination and isReady reflects
    ///        the outcome of whichever branch actually ran.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:64-67, 237-248
    func test_characterizes_afterStreamEnds_instanceIsNotReadyWithNoFurtherEvent__ODC_0015() async throws {
        try requireSimulatorStub()
        let url = try SyntheticModelFixture.make(named: "generic-probe.gguf")
        let profile = try ModelProfile(filePath: url.path)
        let instance = LlamaInstance(profile: profile, settings: .balanced, predictionConfig: .balanced)

        let stream = instance.initialize()
        let recorder = StreamRecorder<LoadProgress>()
        let recording = await recorder.record(stream, timeout: Self.loadingStreamTimeout)

        XCTAssertTrue(recording.terminated)
        XCTAssertFalse(instance.isReady)
    }
}

final class D3FailurePathTerminationCharacterizationTests: XCTestCase {

    private static let loadingStreamTimeout: Duration = .seconds(20)

    override func tearDown() {
        SyntheticModelFixture.removeAll()
        super.tearDown()
    }

    /// CHARACTERIZATION D3 (ODC-0012)
    /// Today: the failure-path loading stream terminates because cleanup()
    ///        (LlamaInstance.swift:246-247) finishes loadingContinuation, not
    ///        because the gate at publishProgress (:583, unsatisfiable per
    ///        C-D3-1) ever fires. Asserted here by the absence of any terminal
    ///        event before the stream ends, on the stub's deterministic
    ///        failure path.
    /// Should be: the gate fires and a terminal event precedes termination.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:580-587
    func test_characterizes_failurePathStream_terminatesWithoutTheGateFiring__ODC_0012() async throws {
        try requireSimulatorStub()
        let url = try SyntheticModelFixture.make(named: "generic-probe.gguf")
        let profile = try ModelProfile(filePath: url.path)
        let instance = LlamaInstance(profile: profile, settings: .balanced, predictionConfig: .balanced)

        let stream = instance.initialize()
        let recorder = StreamRecorder<LoadProgress>()
        let recording = await recorder.record(stream, timeout: Self.loadingStreamTimeout)

        XCTAssertTrue(recording.terminated, "the stream ends via cleanup(), not via the gate")
        let sawTerminal = recording.events.contains { progress in
            if case .ready = progress { return true }
            if case .failed = progress { return true }
            return false
        }
        XCTAssertFalse(sawTerminal, "no .ready or .failed precedes termination")
    }
}

final class D4SimulatorStubCharacterizationTests: XCTestCase {

    override func tearDown() {
        SyntheticModelFixture.removeAll()
        super.tearDown()
    }

    /// CHARACTERIZATION D4 (ODC-0013)
    /// Today: on SFC-B, LlamaBridge.loadModel fails for every file that
    ///        reaches it, because the linked slice is the 51-symbol stub
    ///        whose `_llama_load_model_from_file` returns null
    ///        unconditionally (spec N3). This is the runtime face of D4: the
    ///        package declares .macOS(.v14) with no macOS slice and links a
    ///        non-functional stub on the simulator.
    /// Should be: on a real slice this case skips
    ///        (SKIP[requires-simulator-stub]); loadModel succeeds for a valid
    ///        model file once ODC-0013 lands a functional slice.
    /// Evidence: Sources/OnDeviceCatalyst/API Bridge/LlamaBridge.swift:42-62
    func test_characterizes_llamaBridgeLoadModel_failsForEveryFileOnTheStub__ODC_0013() throws {
        try requireSimulatorStub()
        let url = try SyntheticModelFixture.make(named: "generic-probe.gguf")

        XCTAssertThrowsError(try LlamaBridge.loadModel(path: url.path, settings: .balanced)) { error in
            XCTAssertTrue(error is CatalystError)
        }
    }
}
