// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore

/// Per-stage wall-clock breakdown for a single `SwitchcraftStore.search()`
/// call (issue #148).
///
/// Every field except `cascadeCompactionOccurred` is optional: a stage
/// that had not yet run when the call returned (success) or threw
/// (timeout) is honestly represented as `nil`, not a zero-duration
/// sentinel. Populated incrementally as `search()` progresses through its
/// checkpoints, so a `SwitchcraftStoreError.searchTimedOut` thrown partway
/// through still carries whatever prefix of stages completed.
public struct SearchTiming: Sendable, Equatable {
    /// Wall time inside `flushAndClearPending()` / `indexer.flush()`.
    public var flushDuration: Duration?
    /// Whether the flush triggered an L0 cascade compaction.
    public var cascadeCompactionOccurred: Bool = false
    /// Wall time attributable to the cascade compaction, when one
    /// occurred. Equal to `flushDuration` (the whole flush call), since
    /// `Indexer.performFlush()` has no internal boundary separating
    /// cascade-only time from ordinary flush time.
    public var cascadeCompactionDuration: Duration?
    /// Wall time in the `embedder.encodeQuery()` call.
    public var embedDuration: Duration?
    /// Wall time in the XTR/vector retrieval sub-stage of `searchHybrid`.
    public var vectorDuration: Duration?
    /// Wall time in the BM25/FTS sub-stage of `searchHybrid`.
    public var bm25Duration: Duration?
    /// Wall time in the RRF fusion/union/sort sub-stage of `searchHybrid`.
    public var mergeDuration: Duration?

    public init(
        flushDuration: Duration? = nil,
        cascadeCompactionOccurred: Bool = false,
        cascadeCompactionDuration: Duration? = nil,
        embedDuration: Duration? = nil,
        vectorDuration: Duration? = nil,
        bm25Duration: Duration? = nil,
        mergeDuration: Duration? = nil
    ) {
        self.flushDuration = flushDuration
        self.cascadeCompactionOccurred = cascadeCompactionOccurred
        self.cascadeCompactionDuration = cascadeCompactionDuration
        self.embedDuration = embedDuration
        self.vectorDuration = vectorDuration
        self.bm25Duration = bm25Duration
        self.mergeDuration = mergeDuration
    }
}

/// `SwitchcraftStore.search()`'s return value: the fused hits plus the
/// per-stage timing breakdown for that call (issue #148).
public struct SearchResult: Sendable {
    /// Hits sorted by `(score DESC, uuid ASC)` — see `HybridHit`.
    public let hits: [HybridHit]
    /// Per-stage timing breakdown for this call.
    public let timing: SearchTiming

    public init(hits: [HybridHit], timing: SearchTiming) {
        self.hits = hits
        self.timing = timing
    }
}
