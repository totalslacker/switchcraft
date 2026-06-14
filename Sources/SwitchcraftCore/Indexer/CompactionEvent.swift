// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A snapshot of the compaction that is about to be persisted, delivered
/// to `SwitchcraftStore.onCompactionEvent` once per `flush()` that
/// produces at least one output generation.
///
/// All field values reflect the **final, post-recovery** state of the
/// compaction (see ADR 030). If the mid-operation divergence recovery
/// absorbs additional generations, `inputBytes` and `inputSegments` will
/// be larger than the preliminary cascade-walk estimates.
public struct CompactionEvent: Sendable {

    /// Reason the compaction was initiated.
    public enum Trigger: String, Sendable {
        /// A level's capacity threshold was exceeded, causing the pending
        /// buffer to be promoted into higher LSM levels.
        case cascade

        /// An explicit `Indexer.flush()` / `SwitchcraftStore.index()` call
        /// that produced a new L0 generation without cascading to higher levels.
        case manual
    }

    /// Total byte size of the embeddings being merged. Equal to
    /// `inputSegments`' total embedding count × embedding dimensions × 4
    /// (Float32 bytes), after any mid-operation divergence recovery.
    public let inputBytes: UInt64

    /// Reason this compaction was initiated.
    public let trigger: Trigger

    /// Number of LSM generations (including the pending L0 buffer) merged
    /// into the new generation. Always ≥ 1. Updated by divergence recovery
    /// if additional generations were absorbed.
    public let inputSegments: Int

    /// Wall-clock timestamp captured at the start of `performFlush()`,
    /// before k-means training begins.
    public let startedAt: Date
}
