//
//  StreamContractCharacterizationTests.swift
//  OnDeviceCatalystTests
//
//  ODC-0004, R1: D2 (LlamaInstance.performGeneration yields a second, always
//  .natural completion after generateTokens already yielded the real reason;
//  see Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:384-386 and
//  the five emit sites in generateTokens, :466-536).
//
//  These cases synthesize the exact chunk *shape* v2 emits -- a real-reason
//  completion followed by an always-.natural completion -- and exercise the
//  consumer-side contract in Sources/OnDeviceCatalyst/Chat System/StreamResponse.swift
//  against it. No llama return value is consumed; generation itself (the
//  producer side, C-D2-3/C-D2-4) needs real inference and is R3.
//

import XCTest
@testable import OnDeviceCatalyst

final class D2StreamContractCharacterizationTests: XCTestCase {

    /// Builds the chunk sequence v2's generator actually emits when generation
    /// stops for `realReason`: one content chunk, a completion carrying the
    /// real reason (generateTokens' own emit site), and a second completion
    /// whose reason is unconditionally `.natural` (performGeneration's emit
    /// site, which fires regardless of how generateTokens returned).
    private func makeD2ChunkStream(realReason: CompletionReason) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.content("partial output"))
            continuation.yield(.completion(reason: realReason))
            continuation.yield(.completion(reason: .natural))
            continuation.finish()
        }
    }

    /// CHARACTERIZATION D2 (ODC-0011)
    /// Today: performGeneration yields a second completion whose reason is
    ///        always .natural, after generateTokens already yielded the real
    ///        reason. A collector that drains every chunk (does not break on
    ///        the first isComplete chunk) ends up recording the second,
    ///        always-.natural reason, and the real reason is lost.
    /// Should be: exactly one completion chunk exists, carrying the real
    ///        reason, so draining and breaking consumers agree.
    /// Evidence: Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:384-386
    func test_characterizes_streamingResponse_drainedCollector_reportsNaturalOverRealReason__ODC_0011() async throws {
        let stream = makeD2ChunkStream(realReason: .maxTokensReached)

        var drained = StreamingResponse()
        for try await chunk in stream {
            drained.addChunk(chunk)
        }

        XCTAssertEqual(drained.completionReason, .natural)
        XCTAssertNotEqual(drained.completionReason, .maxTokensReached, "the real reason was overwritten by the spurious second completion")
    }

    /// CHARACTERIZATION D2 (ODC-0011)
    /// Today: collectResponse() and collectContent() both break on the first
    ///        completion chunk they see, so they observe the real reason from
    ///        generateTokens' emit site, never the spurious .natural second
    ///        completion. A consumer that breaks and a consumer that drains
    ///        (see the sibling case above) therefore disagree about why
    ///        generation stopped, for the identical stream.
    /// Should be: both consumers report the same, real reason, because only
    ///        one completion chunk exists.
    /// Evidence: Sources/OnDeviceCatalyst/Chat System/StreamResponse.swift:275-300,
    ///           Sources/OnDeviceCatalyst/Core Engine/LlamaInstance.swift:384-386
    func test_characterizes_collectResponse_breakingCollector_disagreesWithDrainedCollector__ODC_0011() async throws {
        let breakingStream = makeD2ChunkStream(realReason: .maxTokensReached)
        let broken = try await breakingStream.collectResponse()
        XCTAssertEqual(broken.completionReason, .maxTokensReached, "a breaking collector sees the real reason")

        let drainingStream = makeD2ChunkStream(realReason: .maxTokensReached)
        var drained = StreamingResponse()
        for try await chunk in drainingStream {
            drained.addChunk(chunk)
        }
        XCTAssertEqual(drained.completionReason, .natural, "a draining collector sees the spurious reason")

        XCTAssertNotEqual(
            broken.completionReason,
            drained.completionReason,
            "two consumers of the identically-shaped stream disagree about the completion reason"
        )
    }

    /// CHARACTERIZATION D2 (ODC-0011)
    /// Today: collectContent() breaks on the first completion chunk exactly
    ///        like collectResponse() does (same "if chunk.isComplete { break }"
    ///        shape), so it never observes the spurious second, always-.natural
    ///        completion performGeneration appends.
    /// Should be: unchanged in observable shape once D2 is fixed, because only
    ///        one completion chunk will exist to break on.
    /// Evidence: Sources/OnDeviceCatalyst/Chat System/StreamResponse.swift:288-300
    func test_characterizes_collectContent_breaksOnFirstCompletion__ODC_0011() async throws {
        let stream = makeD2ChunkStream(realReason: .maxTokensReached)
        let content = try await stream.collectContent()
        XCTAssertEqual(content, "partial output", "collectContent stops accumulating at the first completion chunk")
    }
}
