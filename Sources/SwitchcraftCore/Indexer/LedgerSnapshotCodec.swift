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
/// **v2 (issue #137)**: the payload encodes the full ledger
/// `[Int64: [Indexer.LedgerToken]]` — a per-token tag distinguishes
/// materialized rows (full-precision `[Float]`, copied verbatim) from
/// bucket-refs (`genID`/`bucketID`/`pairOffset`, copied verbatim). This
/// means `writeLedgerSnapshot()` is a pure in-memory encode of whatever the
/// ledger already holds — it never needs to decode a bucket-ref back into
/// floats just to snapshot it, which would reintroduce the O(pairs×dims)
/// cost this issue removes from startup, now on the much-hotter flush path.
/// A snapshot decoded back via `decode(_:dims:)` reproduces the *exact*
/// mixed representation the ledger had at encode time — no precision loss
/// beyond whatever the ledger already carried (materialized rows stay
/// full-precision; bucket-refs stay lossy-only-on-eventual-materialization,
/// same as the live ledger). See ADR 034 (amended) and ADR 036.
///
/// Layout (all integers little-endian):
///   - `UInt32`  magic (`0xFFFF_FFFF` — see `unsupportedVersion` below)
///   - `UInt8`   version (`2`)
///   - `UInt32`  chunkCount
///   - repeated chunkCount times, chunks sorted by ascending chunkID:
///       - `Int64`   chunkID
///       - `UInt32`  tokenCount (rows for this chunk)
///       - repeated tokenCount times, one token after another in ledger order:
///           - `UInt8` tag: `0` = materialized, `1` = bucketRef
///           - tag `0`: dims × `Float32` (little-endian bit patterns)
///           - tag `1`: `Int64` genID, `Int64` bucketID, `Int64` pairOffset
///
/// `dims` is carried out-of-band on the `LedgerSnapshotRecord` (an empty
/// ledger cannot infer it), so it is *not* repeated per row here.
///
/// **Version history**: v1 (issue #136) began directly with the `UInt32`
/// chunkCount — no magic, no version byte, no per-token tag (every row was
/// unconditionally a materialized `[Float]`). `decode(_:dims:)` rejects any
/// payload whose first 4 bytes aren't the v2 magic, which a v1 payload's
/// chunk count can only match by coincidence for an implausible ~4-billion
/// chunk snapshot — for any real payload this is unambiguous. The caller
/// (`loadFromSnapshotIfValid`) treats any decode failure as "snapshot
/// unusable" and falls back to the full rehydration walk, so a v1 snapshot
/// left over from before this upgrade fails closed rather than being
/// misparsed.
enum LedgerSnapshotCodec {

    enum Error: Swift.Error, Sendable, Equatable {
        /// The payload was truncated or otherwise structurally invalid.
        case corruptPayload(String)
        /// The payload's magic/version header doesn't match what this
        /// build of `LedgerSnapshotCodec` writes — most likely a v1
        /// (pre-issue-#137) snapshot left over from before an upgrade.
        case unsupportedVersion(UInt8)
    }

    /// Sentinel first 4 bytes distinguishing the v2 (bucket-ref-aware)
    /// payload format from the v1 (issue #136) format, which began directly
    /// with a `UInt32` chunk count. No real chunk count can plausibly reach
    /// this value, so a v1 payload is always safely rejected rather than
    /// misparsed as v2.
    private static let magic: UInt32 = 0xFFFF_FFFF
    private static let version: UInt8 = 2

    private static let tagMaterialized: UInt8 = 0
    private static let tagBucketRef: UInt8 = 1

    /// Exact wire-format byte size for `encode(_:dims:)`'s output, computed
    /// without allocating the output buffer. Used to `reserveCapacity` (and,
    /// post-issue-#144, to preallocate the exact-size bulk buffer) so the
    /// encoder never pays repeated-reallocation cost on a large ledger,
    /// mirroring `decode()`'s pre-sizing of its own output collections.
    private static func encodedByteCount(_ ledger: [Int64: [Indexer.LedgerToken]], dims: Int) -> Int {
        let rowBytes = dims * 4
        // magic (4) + version (1) + chunkCount (4)
        var size = 9
        for tokens in ledger.values {
            // chunkID (8) + tokenCount (4)
            size += 12
            for token in tokens {
                switch token {
                case .materialized:
                    size += 1 + rowBytes
                case .bucketRef:
                    size += 1 + 24
                }
            }
        }
        return size
    }

    /// Number of chunks encoded between each yield/progress checkpoint
    /// (issue #144, REQ-4/REQ-5). Picked so a 100K-chunk ledger yields on
    /// the order of ~200 times — frequent enough that a large encode
    /// cannot starve the cooperative thread pool for more than a few
    /// hundred chunks' worth of work at a stretch, infrequent enough that
    /// the `await`/callback overhead stays negligible next to the encode
    /// work itself.
    private static let yieldInterval = 512

    /// Encode the ledger into a payload blob. Chunks are emitted in ascending
    /// chunkID order so the encoding is deterministic for a given ledger.
    ///
    /// Builds directly into a preallocated `[UInt8]` buffer with indexed
    /// writes rather than `Data.append` (issue #144, REQ-3): at production
    /// scale (~11.1M embeddings) the append-based version made on the order
    /// of hundreds of millions of individual single-byte `Data.append`
    /// calls, each paying its own COW/bounds-check overhead. Computing the
    /// exact size up front and writing at known offsets removes that
    /// per-call overhead entirely.
    ///
    /// `async` so it can periodically `await Task.yield()` (REQ-4): this
    /// runs synchronously on the `Indexer` actor with no other `await`
    /// points, so an unyielding multi-second encode over a large ledger
    /// would otherwise occupy one of the process's cooperative-pool worker
    /// threads for the whole duration, risking starvation of unrelated
    /// concurrent work elsewhere in the host app. Periodic yielding is only
    /// safe because every caller holds the `flushInProgress` leader gate for
    /// the duration of the call (see `Indexer.writeLedgerSnapshot()`) — no
    /// concurrent `add()`/`removeFromLedger()` can mutate `ledger` between
    /// yields. Any future caller of `encode()` must hold that same gate.
    ///
    /// `onProgress`, if supplied, is invoked at the same checkpoints with
    /// chunks/bytes encoded so far (REQ-5), from this function's caller's
    /// execution context (the `Indexer` actor) — a callback that needs to
    /// hop elsewhere must do so itself.
    ///
    /// - Precondition: every `.materialized` row has exactly `dims` elements.
    static func encode(
        _ ledger: [Int64: [Indexer.LedgerToken]],
        dims: Int,
        onProgress: (@Sendable (SnapshotProgress) async -> Void)? = nil
    ) async -> Data {
        precondition(dims > 0, "dims must be positive")
        let sortedChunkIDs = ledger.keys.sorted()
        let totalSize = encodedByteCount(ledger, dims: dims)

        var buffer = [UInt8](repeating: 0, count: totalSize)
        var offset = 0

        writeUInt32LE(&buffer, offset, magic); offset += 4
        buffer[offset] = version; offset += 1
        writeUInt32LE(&buffer, offset, UInt32(sortedChunkIDs.count)); offset += 4

        for (i, chunkID) in sortedChunkIDs.enumerated() {
            let tokens = ledger[chunkID]!
            writeInt64LE(&buffer, offset, chunkID); offset += 8
            writeUInt32LE(&buffer, offset, UInt32(tokens.count)); offset += 4
            for token in tokens {
                switch token {
                case .materialized(let row):
                    precondition(row.count == dims,
                                 "ledger row for chunk \(chunkID) has \(row.count) elements; expected dims \(dims)")
                    buffer[offset] = tagMaterialized; offset += 1
                    for v in row {
                        writeFloat32LE(&buffer, offset, v); offset += 4
                    }
                case .bucketRef(let genID, let bucketID, let pairOffset):
                    buffer[offset] = tagBucketRef; offset += 1
                    writeInt64LE(&buffer, offset, genID); offset += 8
                    writeInt64LE(&buffer, offset, bucketID); offset += 8
                    writeInt64LE(&buffer, offset, Int64(pairOffset)); offset += 8
                }
            }

            let chunksEncoded = i + 1
            if chunksEncoded % yieldInterval == 0 {
                if let onProgress {
                    await onProgress(SnapshotProgress(
                        chunksEncoded: chunksEncoded, totalChunks: sortedChunkIDs.count, bytesEncoded: offset
                    ))
                }
                await Task.yield()
            }
        }

        if let onProgress, !sortedChunkIDs.isEmpty {
            await onProgress(SnapshotProgress(
                chunksEncoded: sortedChunkIDs.count, totalChunks: sortedChunkIDs.count, bytesEncoded: offset
            ))
        }

        precondition(offset == totalSize, "encoded size mismatch: wrote \(offset) bytes, expected \(totalSize)")
        return Data(buffer)
    }

    /// Decode a payload blob back into a ledger. Throws
    /// `Error.unsupportedVersion` if the magic/version header doesn't
    /// match, or `Error.corruptPayload` if the blob is truncated or its
    /// declared counts overrun the buffer — the caller treats any throw as
    /// "snapshot unusable, fall back to full rehydration".
    static func decode(_ data: Data, dims: Int) throws -> [Int64: [Indexer.LedgerToken]] {
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

        try need(4, "magic")
        let readMagic = readUInt32LE(bytes, offset); offset += 4
        guard readMagic == magic else {
            throw Error.unsupportedVersion(0)
        }
        try need(1, "version")
        let readVersion = bytes[offset]; offset += 1
        guard readVersion == version else {
            throw Error.unsupportedVersion(readVersion)
        }

        try need(4, "chunkCount")
        let chunkCount = readUInt32LE(bytes, offset); offset += 4

        var ledger: [Int64: [Indexer.LedgerToken]] = [:]
        ledger.reserveCapacity(Int(chunkCount))
        let rowBytes = dims * 4

        for _ in 0..<chunkCount {
            try need(8, "chunkID")
            let chunkID = readInt64LE(bytes, offset); offset += 8
            try need(4, "tokenCount")
            let tokenCount = Int(readUInt32LE(bytes, offset)); offset += 4

            var tokens: [Indexer.LedgerToken] = []
            tokens.reserveCapacity(tokenCount)
            for _ in 0..<tokenCount {
                try need(1, "chunk \(chunkID) token tag")
                let tag = bytes[offset]; offset += 1
                switch tag {
                case tagMaterialized:
                    try need(rowBytes, "chunk \(chunkID) materialized row")
                    var row = [Float]()
                    row.reserveCapacity(dims)
                    for _ in 0..<dims {
                        row.append(readFloat32LE(bytes, offset)); offset += 4
                    }
                    tokens.append(.materialized(row))
                case tagBucketRef:
                    try need(24, "chunk \(chunkID) bucketRef")
                    let genID = readInt64LE(bytes, offset); offset += 8
                    let bucketID = readInt64LE(bytes, offset); offset += 8
                    let pairOffset = Int(readInt64LE(bytes, offset)); offset += 8
                    tokens.append(.bucketRef(genID: genID, bucketID: bucketID, pairOffset: pairOffset))
                default:
                    throw Error.corruptPayload("unknown token tag \(tag) for chunk \(chunkID) at offset \(offset - 1)")
                }
            }
            ledger[chunkID] = tokens
        }

        if offset != bytes.count {
            throw Error.corruptPayload(
                "trailing bytes: consumed \(offset) of \(bytes.count)"
            )
        }
        return ledger
    }

    // MARK: - LE primitives

    /// Write a little-endian `UInt32` at `buffer[offset..<offset+4]`.
    /// Indexed writes (vs. `Data.append`) avoid a per-call COW/bounds-check
    /// for each of the millions of integers a large ledger encodes (REQ-3).
    @inline(__always)
    private static func writeUInt32LE(_ buffer: inout [UInt8], _ offset: Int, _ v: UInt32) {
        buffer[offset] = UInt8(v & 0xFF)
        buffer[offset + 1] = UInt8((v >> 8) & 0xFF)
        buffer[offset + 2] = UInt8((v >> 16) & 0xFF)
        buffer[offset + 3] = UInt8((v >> 24) & 0xFF)
    }

    @inline(__always)
    private static func writeInt64LE(_ buffer: inout [UInt8], _ offset: Int, _ v: Int64) {
        let bits = UInt64(bitPattern: v)
        for k in 0..<8 {
            buffer[offset + k] = UInt8((bits >> UInt64(k * 8)) & 0xFF)
        }
    }

    @inline(__always)
    private static func writeFloat32LE(_ buffer: inout [UInt8], _ offset: Int, _ v: Float) {
        writeUInt32LE(&buffer, offset, v.bitPattern)
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
