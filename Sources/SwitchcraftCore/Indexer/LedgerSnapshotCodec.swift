// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A cheap staleness/corruption fingerprint of on-disk storage state.
///
/// Computed from metadata that is already available to `Indexer.init` and
/// `Indexer.performFlush()` without decoding any bucket blobs:
/// `storage.chunkCount()` (O(1) `COUNT(*)`) plus fields already present on
/// `storage.generations()`. Compared for exact equality against the values
/// captured when a `LedgerSnapshotRecord` was written; any mismatch means the
/// snapshot is stale (storage changed out-of-band, e.g. via `vacuum()`) or
/// corrupt, and the caller falls back to the full rehydration walk.
///
/// This is deliberately *not* a cryptographic integrity guarantee — it is a
/// fast, coarse check. Correctness is backstopped by the full-walk fallback
/// on any mismatch. See ADR 034.
public struct LedgerSnapshotFingerprint: Sendable, Hashable {
    public let chunkCount: Int
    public let maxChunkID: Int64
    public let totalEmbeddings: Int
    public let maxGenerationID: Int64
    public let generationCount: Int

    public init(
        chunkCount: Int,
        maxChunkID: Int64,
        totalEmbeddings: Int,
        maxGenerationID: Int64,
        generationCount: Int
    ) {
        self.chunkCount = chunkCount
        self.maxChunkID = maxChunkID
        self.totalEmbeddings = totalEmbeddings
        self.maxGenerationID = maxGenerationID
        self.generationCount = generationCount
    }

    /// Derive the fingerprint from `chunkCount` plus the already-fetched
    /// generation list. "Max chunk id" is derived as `max(gen.maxChunkID)`
    /// rather than an independent storage query — no cheap such query exists,
    /// and it is correct at both snapshot write points (post-flush and clean
    /// shutdown), where every committed chunk belongs to some generation.
    public static func compute(
        chunkCount: Int,
        generations: [GenerationRecord]
    ) -> LedgerSnapshotFingerprint {
        var maxChunkID: Int64 = 0
        var totalEmbeddings = 0
        var maxGenerationID: Int64 = 0
        for gen in generations {
            maxChunkID = max(maxChunkID, gen.maxChunkID)
            totalEmbeddings += gen.numEmbeddings
            maxGenerationID = max(maxGenerationID, gen.id)
        }
        return LedgerSnapshotFingerprint(
            chunkCount: chunkCount,
            maxChunkID: maxChunkID,
            totalEmbeddings: totalEmbeddings,
            maxGenerationID: maxGenerationID,
            generationCount: generations.count
        )
    }

    /// The fingerprint captured in a persisted snapshot record.
    public static func of(_ record: LedgerSnapshotRecord) -> LedgerSnapshotFingerprint {
        LedgerSnapshotFingerprint(
            chunkCount: record.chunkCount,
            maxChunkID: record.maxChunkID,
            totalEmbeddings: record.totalEmbeddings,
            maxGenerationID: record.maxGenerationID,
            generationCount: record.generationCount
        )
    }
}

/// Wire-format codec for a `LedgerSnapshotRecord.payload` blob.
///
/// The payload encodes the full ledger `[Int64: [[Float]]]` as a flat,
/// self-describing byte stream so that decoding reconstructs the *exact*
/// in-memory structure the full rehydration walk would have produced — same
/// `[Float]` values, same per-chunk token ordering. Bit-for-bit parity here
/// is required or NFCorpus retrieval results would silently drift (see
/// ADR 034 / issue #136 requirement 2).
///
/// Layout (all integers little-endian):
///   - `UInt32`  chunkCount
///   - repeated chunkCount times, chunks sorted by ascending chunkID:
///       - `Int64`   chunkID
///       - `UInt32`  tokenCount (rows for this chunk)
///       - tokenCount × dims × `Float32` (little-endian bit patterns), one
///         row after another in ledger order
///
/// `dims` is carried out-of-band on the `LedgerSnapshotRecord` (an empty
/// ledger cannot infer it), so it is *not* repeated per row here.
public enum LedgerSnapshotCodec {

    public enum Error: Swift.Error, Sendable, Equatable {
        /// The payload was truncated or otherwise structurally invalid.
        case corruptPayload(String)
    }

    /// Encode the ledger into a payload blob. Chunks are emitted in ascending
    /// chunkID order so the encoding is deterministic for a given ledger.
    ///
    /// - Precondition: every row in every chunk has exactly `dims` elements.
    public static func encode(_ ledger: [Int64: [[Float]]], dims: Int) -> Data {
        precondition(dims > 0, "dims must be positive")
        let sortedChunkIDs = ledger.keys.sorted()

        // Pre-size: 4 (count) + per chunk 8 (id) + 4 (tokenCount) + rows*dims*4.
        var capacity = 4
        for chunkID in sortedChunkIDs {
            let rows = ledger[chunkID]!
            capacity += 8 + 4 + rows.count * dims * 4
        }
        var out = Data(capacity: capacity)

        appendUInt32LE(&out, UInt32(sortedChunkIDs.count))
        for chunkID in sortedChunkIDs {
            let rows = ledger[chunkID]!
            appendInt64LE(&out, chunkID)
            appendUInt32LE(&out, UInt32(rows.count))
            for row in rows {
                precondition(row.count == dims,
                             "ledger row for chunk \(chunkID) has \(row.count) elements; expected dims \(dims)")
                for v in row {
                    appendFloat32LE(&out, v)
                }
            }
        }
        return out
    }

    /// Decode a payload blob back into a ledger. Throws
    /// `Error.corruptPayload` if the blob is truncated or its declared
    /// counts overrun the buffer — the caller treats a throw as "snapshot
    /// unusable, fall back to full rehydration".
    public static func decode(_ data: Data, dims: Int) throws -> [Int64: [[Float]]] {
        guard dims > 0 else {
            throw Error.corruptPayload("dims must be positive, got \(dims)")
        }
        // Work over a contiguous copy so index math is startIndex-relative-safe.
        let bytes = [UInt8](data)
        var offset = 0

        func need(_ n: Int, _ what: String) throws {
            if offset + n > bytes.count {
                throw Error.corruptPayload(
                    "unexpected end of payload reading \(what): need \(n) bytes at offset \(offset), have \(bytes.count)"
                )
            }
        }

        try need(4, "chunkCount")
        let chunkCount = readUInt32LE(bytes, offset); offset += 4

        var ledger: [Int64: [[Float]]] = [:]
        ledger.reserveCapacity(Int(chunkCount))
        let rowBytes = dims * 4

        for _ in 0..<chunkCount {
            try need(8, "chunkID")
            let chunkID = readInt64LE(bytes, offset); offset += 8
            try need(4, "tokenCount")
            let tokenCount = Int(readUInt32LE(bytes, offset)); offset += 4

            try need(tokenCount * rowBytes, "chunk \(chunkID) rows")
            var rows: [[Float]] = []
            rows.reserveCapacity(tokenCount)
            for _ in 0..<tokenCount {
                var row = [Float]()
                row.reserveCapacity(dims)
                for _ in 0..<dims {
                    row.append(readFloat32LE(bytes, offset)); offset += 4
                }
                rows.append(row)
            }
            ledger[chunkID] = rows
        }

        if offset != bytes.count {
            throw Error.corruptPayload(
                "trailing bytes: consumed \(offset) of \(bytes.count)"
            )
        }
        return ledger
    }

    // MARK: - LE primitives

    private static func appendUInt32LE(_ out: inout Data, _ v: UInt32) {
        out.append(UInt8(v & 0xFF))
        out.append(UInt8((v >> 8) & 0xFF))
        out.append(UInt8((v >> 16) & 0xFF))
        out.append(UInt8((v >> 24) & 0xFF))
    }

    private static func appendInt64LE(_ out: inout Data, _ v: Int64) {
        let bits = UInt64(bitPattern: v)
        for shift in stride(from: 0, through: 56, by: 8) {
            out.append(UInt8((bits >> UInt64(shift)) & 0xFF))
        }
    }

    private static func appendFloat32LE(_ out: inout Data, _ v: Float) {
        appendUInt32LE(&out, v.bitPattern)
    }

    private static func readUInt32LE(_ b: [UInt8], _ i: Int) -> UInt32 {
        UInt32(b[i])
            | (UInt32(b[i + 1]) << 8)
            | (UInt32(b[i + 2]) << 16)
            | (UInt32(b[i + 3]) << 24)
    }

    private static func readInt64LE(_ b: [UInt8], _ i: Int) -> Int64 {
        var bits: UInt64 = 0
        for k in 0..<8 {
            bits |= UInt64(b[i + k]) << UInt64(k * 8)
        }
        return Int64(bitPattern: bits)
    }

    private static func readFloat32LE(_ b: [UInt8], _ i: Int) -> Float {
        Float(bitPattern: readUInt32LE(b, i))
    }
}
