// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore

/// A test-only `Embedder` wrapping `MockEmbedder`, injecting a
/// configurable `Task.sleep` before `encode(_:)` and `encodeQuery(_:minSurfaceFormLength:)`.
///
/// Used by `SearchTimingTests` (issue #148) to prove that a slow embed
/// stage is attributed to `embedDuration` alone and doesn't inflate
/// `flushDuration` or the `searchHybrid` sub-stage durations, without a
/// hardcoded seconds-scale sleep.
struct DelayedEmbedder: Embedder, Sendable {
    private let inner: MockEmbedder
    let delay: Duration

    var dims: Int { inner.dims }
    var modelIdentifier: String { inner.modelIdentifier }
    var maxInputTokens: Int { inner.maxInputTokens }

    init(dims: Int = 32, delay: Duration) {
        inner = MockEmbedder(dims: dims)
        self.delay = delay
    }

    func encode(_ text: String) async throws -> [Float] {
        if delay > .zero {
            try await Task.sleep(for: delay)
        }
        return try await inner.encode(text)
    }

    func encodeQuery(_ text: String, minSurfaceFormLength: Int) async throws -> [Float] {
        if delay > .zero {
            try await Task.sleep(for: delay)
        }
        return try await inner.encodeQuery(text, minSurfaceFormLength: minSurfaceFormLength)
    }
}
