// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Per-stage wall-clock breakdown for a single `SearchEngine.searchHybrid`
/// call, captured with `ContinuousClock` at each stage boundary (issue
/// #148). All fields are always fully measured on any successful return —
/// a stage that is skipped by an empty-input fallback (e.g. no query
/// embeddings) still reports a (near-zero) duration for the work actually
/// done in that stage, rather than `nil`.
public struct HybridSearchTiming: Sendable {
    /// Wall time in the vector/XTR retrieval sub-stage.
    public let vectorDuration: Duration
    /// Wall time in the BM25/FTS sub-stage.
    public let bm25Duration: Duration
    /// Wall time in the RRF fusion/union/sort sub-stage.
    public let mergeDuration: Duration

    public init(vectorDuration: Duration, bm25Duration: Duration, mergeDuration: Duration) {
        self.vectorDuration = vectorDuration
        self.bm25Duration = bm25Duration
        self.mergeDuration = mergeDuration
    }
}

/// `SearchEngine.searchHybrid`'s return value: the fused hits plus the
/// per-stage timing breakdown for that call (issue #148).
public struct HybridSearchResult: Sendable {
    /// Hits sorted by `(score DESC, uuid ASC)` — see `HybridHit`.
    public let hits: [HybridHit]
    /// Per-stage timing breakdown for this call.
    public let timing: HybridSearchTiming

    public init(hits: [HybridHit], timing: HybridSearchTiming) {
        self.hits = hits
        self.timing = timing
    }
}
