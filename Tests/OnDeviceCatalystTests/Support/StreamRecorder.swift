//
//  StreamRecorder.swift
//  OnDeviceCatalystTests
//
//  ODC-0004: bounded collection of AsyncStream events. The primitive behind
//  every stream assertion in this suite. Returns both the events observed and
//  whether the stream terminated within the bound, so "terminated with no
//  terminal event" (spec N5) and "never terminated" (D3 success path, R3) are
//  the same call with different expectations.
//
//  Every timeout here is a liveness bound, not a benchmark. See spec
//  "## Benchmarks".
//

import Foundation

public actor StreamRecorder<Element: Sendable> {

    public struct Recording {
        public let events: [Element]
        public let terminated: Bool
    }

    public init() {}

    /// Records events from `stream` until it finishes or `timeout` elapses,
    /// whichever comes first.
    public func record(_ stream: AsyncStream<Element>, timeout: Duration) async -> Recording {
        var events: [Element] = []

        let collector = Task<Bool, Never> {
            for await element in stream {
                events.append(element)
            }
            return true
        }

        let watchdog = Task<Void, Never> {
            try? await Task.sleep(for: timeout)
            collector.cancel()
        }

        let terminated = await collector.value
        watchdog.cancel()

        // If the collector was cancelled before the stream naturally finished,
        // `terminated` above is still `true` because the for-await loop exits
        // on cancellation too. Distinguish the two by racing against the bound.
        return Recording(events: events, terminated: terminated && !collector.isCancelled)
    }
}
