// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Incremental progress during `LedgerSnapshotCodec.encode()`'s per-chunk
/// loop, delivered to an optional `onProgress` callback threaded through
/// `encode(_:dims:onProgress:)` → `writeLedgerSnapshot(onProgress:)` →
/// `persistSnapshot(onProgress:)` → `SwitchcraftStore.shutdown(onSnapshotProgress:)`
/// (issue #144, REQ-5).
///
/// Delivered from the `Indexer` actor's execution context — a callback that
/// hops to another actor/the main thread must do so itself, matching the
/// `setOnCompactionEvent` precedent (ADR 032).
public struct SnapshotProgress: Sendable, Equatable {
    /// Chunks encoded so far, out of `totalChunks`.
    public let chunksEncoded: Int
    /// Total chunks in the ledger being encoded.
    public let totalChunks: Int
    /// Bytes written to the output buffer so far.
    public let bytesEncoded: Int

    public init(chunksEncoded: Int, totalChunks: Int, bytesEncoded: Int) {
        self.chunksEncoded = chunksEncoded
        self.totalChunks = totalChunks
        self.bytesEncoded = bytesEncoded
    }
}

/// Wall-clock/size breakdown of one `Indexer.writeLedgerSnapshot()` call,
/// returned up through `persistSnapshot()` so `SwitchcraftStore.shutdown()`
/// can log the REQ-1 instrumentation lines without `SwitchcraftCore` itself
/// depending on the `switchcraft-adapter-shutdown:`-prefixed log format that
/// the issue specifies (an issue-mandated, not codebase-derived, convention
/// that belongs to the `Switchcraft` module's `shutdown()`).
///
/// All-zero when `writeLedgerSnapshot()` took an early-return path (no
/// `dims` locked in yet, or the write-time consistency guard refused the
/// write — see ADR 034 §5) — neither path calls `LedgerSnapshotCodec.encode`
/// or `storage.saveLedgerSnapshot`.
public struct PersistSnapshotStats: Sendable, Equatable {
    public let encodeSeconds: TimeInterval
    public let writeSeconds: TimeInterval
    public let bytesSerialized: Int
    public let chunkCount: Int

    public init(encodeSeconds: TimeInterval, writeSeconds: TimeInterval, bytesSerialized: Int, chunkCount: Int) {
        self.encodeSeconds = encodeSeconds
        self.writeSeconds = writeSeconds
        self.bytesSerialized = bytesSerialized
        self.chunkCount = chunkCount
    }

    static let zero = PersistSnapshotStats(encodeSeconds: 0, writeSeconds: 0, bytesSerialized: 0, chunkCount: 0)
}
