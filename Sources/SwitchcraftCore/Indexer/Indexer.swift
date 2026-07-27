// SPDX-License-Identifier: Apache-2.0
import Foundation
import os

private let indexerLogger = Logger(subsystem: "com.switchcraft.core", category: "Indexer")

extension Indexer.Error: Equatable {
    /// Hand-written rather than derived: `bucketRefUnresolvable.cause` is a
    /// diagnostic-only `String?` capturing an underlying `Swift.Error`'s
    /// description. Comparing it would make test assertions like
    /// `#expect(throws: Indexer.Error.bucketRefUnresolvable(...))` brittle to
    /// incidental wording rather than the error's actual identity
    /// (`chunkID`/`bucketID`), so `cause` is excluded from equality.
    public static func == (lhs: Indexer.Error, rhs: Indexer.Error) -> Bool {
        switch (lhs, rhs) {
        case let (.ledgerOutOfSync(l1, l2), .ledgerOutOfSync(r1, r2)):
            return l1 == r1 && l2 == r2
        case let (.rehydrationConflict(l), .rehydrationConflict(r)):
            return l == r
        case let (.rehydrationBucketCorrupt(l), .rehydrationBucketCorrupt(r)):
            return l == r
        case let (.bucketRefUnresolvable(lc, lb, _), .bucketRefUnresolvable(rc, rb, _)):
            return lc == rc && lb == rb
        default:
            return false
        }
    }
}

/// LSM-tree token-embedding index over any `SwitchcraftStorage`.
///
/// `add` buffers per-token embeddings for one chunk. `flush` runs
/// k-means on the buffered + lower-level embeddings, persists one
/// `GenerationRecord` plus k `BucketRecord`s, and recursively cascades
/// into higher levels if their per-level capacity is exceeded.
/// `clearIndex` wipes generations and buckets without touching documents
/// or chunks (must NOT call `storage.clear()`, which removes those too).
///
/// Per-bucket invariants the (future) Search pipeline relies on:
///  1. `bucket.center` is an L2-normalised `Float32` vector of length
///     `dims`, little-endian.
///  2. Pairs in `bucket.indices` are sorted ascending by
///     `(chunkID, tokenOffset)` — required for the delta encoding to
///     compress, and the format the Search pipeline scans.
///  3. `bucket.residuals` decodes to exactly `pairs.count * dims`
///     floats, in the same order as the pairs.
///  4. Every token added via `add` is referenced by exactly one bucket
///     in the most recent generation that covers its chunk range.
///  5. `generation.minChunkID` / `generation.maxChunkID` cover all
///     chunk IDs included in the generation.
///
/// The indexer keeps every embedding in an in-memory ledger for
/// cascade re-clustering. On construction the ledger is rehydrated
/// from any existing generations in storage via bucket reconstruction
/// (center + dequantized Q4 residuals), so the indexer survives actor
/// restarts against non-empty persistent storage.
public actor Indexer {

    /// Errors thrown by `Indexer`.
    public enum Error: Swift.Error, Sendable {
        /// The cascade walk computed a row count that does not match the
        /// number of embeddings the in-memory ledger holds for the chunk
        /// range being merged.
        case ledgerOutOfSync(ledgerRows: Int, expected: Int)
        /// The same chunkID was found in two different active generations
        /// during ledger rehydration, which indicates storage corruption.
        /// The expected LSM invariant is that each chunkID appears in at
        /// most one active generation at any given time.
        case rehydrationConflict(chunkID: Int64)
        /// A bucket in storage is structurally invalid and the ledger
        /// cannot be safely reconstructed. Possible causes: the center
        /// blob is empty, residuals length does not equal
        /// `pairs.count * dims`, or two buckets disagree on the
        /// embedding dimension (which must be fixed per database). Thrown
        /// by the cheap init-time integrity walk (issue #137 Requirement
        /// 13): header/codec/size checks only, no residual-value decode.
        case rehydrationBucketCorrupt(generationID: Int64)
        /// On-demand materialization of a `.bucketRef` token failed at
        /// compaction time despite the init-time integrity walk having
        /// passed for that bucket — e.g. a transient disk read error or a
        /// bit-flip surfacing only when the residual contents are actually
        /// decoded. Compaction fails hard on this error; there is no
        /// silent-skip path, since a partial training set would silently
        /// produce subtly wrong k-means output (issue #137 Requirement 14).
        case bucketRefUnresolvable(chunkID: Int64, bucketID: Int64, cause: String?)
    }

    // MARK: - State

    private let storage: any SwitchcraftStorage
    public let config: IndexerConfig

    /// One token embedding as stored in the ledger: either the fully
    /// materialized `[Float]` (memtable rows not yet flushed — there is no
    /// bucket to reference yet) or a lightweight reference into a bucket
    /// already committed to storage. `dims` is not stored per-token since
    /// it's already locked in at the `Indexer` level. See ADR 036.
    enum LedgerToken: Sendable, Equatable {
        case materialized([Float])
        /// `pairOffset` is the token's index into the bucket's decoded
        /// `IndicesCodec.decode(bucket.indices)` array, which is also the
        /// row index into the bucket's decoded residuals (`base =
        /// pairOffset * dims`).
        case bucketRef(genID: Int64, bucketID: Int64, pairOffset: Int)
    }

    /// Every embedding ever added, keyed by chunkID. Token order within
    /// each chunk is the order of the embeddings passed to `add`. Rows are
    /// `.materialized` until the chunk's tokens are first flushed, at which
    /// point `performFlush()` transitions them to `.bucketRef` (issue #137).
    private var ledger: [Int64: [LedgerToken]] = [:]

    /// Vector dimensionality, locked in by the first `add`.
    private var dims: Int?

    /// Number of distinct chunkID conflicts auto-recovered during `init`.
    /// Zero when no conflicts were found or when
    /// `config.rehydrationConflictBehavior == .throwError`.
    public private(set) var recoveredConflictCount: Int = 0

    /// Tracks pending (post-last-flush) state used by the cascade walk.
    private var pendingMinChunkID: Int64?
    private var pendingMaxChunkID: Int64?
    private var pendingCount: Int = 0

    /// Accumulates the count of ledger rows removed by `removeFromLedger()`
    /// since the last successful flush. The cascade walk in `performFlush()`
    /// subtracts this from `pending` so that rehydrated rows cleared by the
    /// R1 partial-orphan path don't inflate `total` and cause a false
    /// `ledgerOutOfSync` (see ADR 029 amendment, issue #132).
    private var removedFromLedgerCount: Int = 0

    /// True while the leader's `performFlush()` body is running. Concurrent
    /// `flush()` callers observe this and join `flushWaiters` instead of
    /// racing past the `pendingCount > 0` gate themselves. See `flush()`.
    private var flushInProgress: Bool = false

    /// Continuations for callers waiting on the leader's in-flight flush.
    /// Holds both concurrent `flush()` callers and `add()` callers that
    /// arrived while a flush was in progress (see `add()` for the
    /// rationale on why `add` waits too). Resumed (with the leader's
    /// success or the leader's thrown error) when the leader's body
    /// returns. Element type matches `withCheckedThrowingContinuation`'s
    /// yielded `CheckedContinuation<_, any Error>`. The `Swift.` prefix
    /// is required because the unqualified `Error` would resolve to the
    /// nested `Indexer.Error` enum in this scope.
    private var flushWaiters: [CheckedContinuation<CompactionEvent?, any Swift.Error>] = []

    // MARK: - Init

    /// Create an indexer over the given storage, rehydrating the in-memory
    /// ledger from any existing generations.
    ///
    /// Rehydration reconstructs per-token embeddings as `center + dequantized
    /// residuals` from bucket data, then populates `ledger` so that a
    /// subsequent `add` + `flush` against non-empty storage succeeds without
    /// `ledgerOutOfSync`. If storage has no committed generations the ledger
    /// starts empty (same behaviour as a fresh index).
    ///
    /// - Throws: `Indexer.Error.rehydrationConflict` if the same chunkID
    ///   appears in two different active generations (storage corruption);
    ///   `Indexer.Error.rehydrationBucketCorrupt` if a bucket has an
    ///   empty center blob, a residuals blob of the wrong length, or a
    ///   dims value that disagrees with other buckets in the database;
    ///   any error thrown by `storage.generations()` or
    ///   `storage.buckets(forGeneration:)`.
    ///
    /// **Breaking change**: this initializer is now `async throws`. The only
    /// in-package call site is `SwitchcraftStore.init`, which is already
    /// `async throws`. External consumers of `SwitchcraftCore` that construct
    /// `Indexer` directly must add `try await`.
    public init(
        storage: any SwitchcraftStorage,
        config: IndexerConfig = .production
    ) async throws {
        self.storage = storage
        self.config = config

        let gens = try await storage.generations()
        guard !gens.isEmpty else { return }

        // Fast path (issue #136 / ADR 034): if a ledger snapshot is present
        // and a cheap fingerprint confirms storage hasn't changed since it was
        // written, load it directly instead of running the O(all embeddings)
        // rehydration walk. Any failure at any step (absent, stale, corrupt)
        // falls through to the full walk below — the crash-recovery safety net
        // is untouched. `recoveredConflictCount` stays 0 on this path: the
        // snapshot represents already-conflict-free state, so there is nothing
        // to recover.
        if try await loadFromSnapshotIfValid(gens: gens) {
            return
        }

        switch config.rehydrationConflictBehavior {
        case .throwError:
            try await rehydrateThrowError(gens: gens)
        case .autoRecover:
            try await rehydrateAutoRecover(gens: gens)
        }
    }

    /// Attempt the snapshot fast path. Returns `true` if the ledger was
    /// populated from a valid, fingerprint-matching snapshot (and the caller
    /// should skip the full rehydration walk); `false` if no usable snapshot
    /// exists and the caller must fall back.
    ///
    /// On a successful load the on-disk snapshot is cleared immediately
    /// (invalidate-on-load, ADR 034): if the process crashes after loading but
    /// before the next successful flush/shutdown, the next startup must fall
    /// back to the full walk rather than trust a snapshot that is now stale
    /// relative to that crash. A fresh snapshot is written again at the next
    /// flush or clean shutdown.
    private func loadFromSnapshotIfValid(gens: [GenerationRecord]) async throws -> Bool {
        guard let record = try await storage.loadLedgerSnapshot() else {
            return false
        }

        // Cheap staleness/corruption check: compare the snapshot's captured
        // fingerprint against freshly-computed storage state.
        let chunkCount = try await storage.chunkCount()
        let current = LedgerSnapshotFingerprint.compute(chunkCount: chunkCount, generations: gens)
        guard LedgerSnapshotFingerprint.of(record) == current else {
            // Stale snapshot (storage mutated out-of-band, e.g. via vacuum()).
            // Clear it so we don't re-check a known-bad snapshot next time, and
            // fall back to the full walk.
            try? await storage.clearLedgerSnapshot()
            return false
        }

        // Decode the payload. A corrupt/truncated blob — or a v1
        // (pre-issue-#137) payload rejected by the version check — throws;
        // treat it as an unusable snapshot and fall back rather than
        // propagating out of init.
        let decoded: [Int64: [LedgerToken]]
        do {
            decoded = try LedgerSnapshotCodec.decode(record.payload, dims: record.dims)
        } catch {
            indexerLogger.warning("Ledger snapshot payload failed to decode (\(String(describing: error), privacy: .public)); falling back to full rehydration")
            try? await storage.clearLedgerSnapshot()
            return false
        }

        // Success: adopt the snapshot state and invalidate the on-disk copy.
        ledger = decoded
        dims = record.dims
        recoveredConflictCount = 0
        didUseSnapshotFastPath = true
        try await storage.clearLedgerSnapshot()
        return true
    }

    /// Cheap, non-residual-decoding half of the Requirement 13 integrity
    /// walk: `bucket.center` decodes to a non-empty `[Float]`, and its
    /// dims agree with any previously-seen bucket in this rehydration pass.
    /// Shared by both rehydration paths. Does not touch `indices` or
    /// `residuals` — callers check those separately (residuals need
    /// `pairs.count`, which requires decoding `indices` first).
    private static func checkedBucketCenter(
        _ bucket: BucketRecord, generationID: Int64, inferredDims: inout Int?
    ) throws -> [Float] {
        let center = Indexer.decodeFloat32LE(bucket.center)
        guard !center.isEmpty else {
            throw Error.rehydrationBucketCorrupt(generationID: generationID)
        }
        if let existing = inferredDims, existing != center.count {
            throw Error.rehydrationBucketCorrupt(generationID: generationID)
        }
        inferredDims = center.count
        return center
    }

    /// The remaining half of the Requirement 13 integrity walk: verify the
    /// `residuals` blob is exactly the byte count Q4Codec would produce for
    /// `pairsCount * dims` values (2 values packed per byte) — a size
    /// check only, not a decode of the residual contents. Catches
    /// truncation/corruption at the same cost as the bucket-ref population
    /// work already required by Requirement 4 (O(pairs), no O(pairs×dims)).
    ///
    /// `Q4Codec.encodeResiduals` requires an even value count (it packs two
    /// 4-bit levels per byte), so `pairsCount * dims` must itself be even
    /// for this bucket to ever have been produced by a legitimate encode.
    /// Checked explicitly before the byte-count comparison: without it, an
    /// odd `pairsCount * dims` (only reachable via corruption — `dims` is
    /// fixed and even for a real corpus, so this implies a corrupt `pairs`
    /// decode) would floor-divide to the same `expectedBytes` as one fewer
    /// residual value, letting a truncated blob slip past this check.
    private static func checkedResidualsSize(
        _ bucket: BucketRecord, pairsCount: Int, dims: Int, generationID: Int64
    ) throws {
        let valueCount = pairsCount * dims
        guard valueCount % 2 == 0 else {
            throw Error.rehydrationBucketCorrupt(generationID: generationID)
        }
        let expectedBytes = valueCount / 2
        guard bucket.residuals.count == expectedBytes else {
            throw Error.rehydrationBucketCorrupt(generationID: generationID)
        }
    }

    /// `.throwError` rehydration path.
    ///
    /// Populates the ledger with `.bucketRef` tokens pointing directly at
    /// bucket metadata (issue #137 Requirement 4) instead of reconstructing
    /// `center + dequantized residuals` per token — the cheap Requirement 13
    /// integrity walk (header/codec/size checks, no residual-value decode)
    /// replaces that reconstruction loop and still surfaces corruption at
    /// store-open time. Throws `rehydrationConflict` on the first detected
    /// chunkID overlap, unchanged from the original implementation.
    private func rehydrateThrowError(gens: [GenerationRecord]) async throws {
        // Accumulate (tokenOffset, genID, bucketID, pairOffset) tuples per
        // chunkID across all buckets of all generations before populating
        // the ledger. Tokens for a single chunkID may be spread across
        // multiple buckets, so the sort step below is required to restore
        // the original tokenOffset order that performFlush() assumes (row
        // index == tokenOffset).
        var accum: [Int64: [(tokenOffset: UInt32, genID: Int64, bucketID: Int64, pairOffset: Int)]] = [:]
        // Maps chunkID → the generationID where it was first seen. Used to
        // detect the storage-corruption case of a chunkID in two active gens.
        var chunkGenMap: [Int64: Int64] = [:]
        var inferredDims: Int? = nil

        for gen in gens {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            for bucket in buckets {
                let center = try Self.checkedBucketCenter(bucket, generationID: gen.id, inferredDims: &inferredDims)
                let d = center.count

                let pairs = try IndicesCodec.decode(bucket.indices)
                try Self.checkedResidualsSize(bucket, pairsCount: pairs.count, dims: d, generationID: gen.id)

                for (i, pair) in pairs.enumerated() {
                    let chunkID = Int64(pair.chunkID)
                    if let seenInGen = chunkGenMap[chunkID], seenInGen != gen.id {
                        throw Error.rehydrationConflict(chunkID: chunkID)
                    }
                    chunkGenMap[chunkID] = gen.id

                    accum[chunkID, default: []].append(
                        (tokenOffset: pair.tokenOffset, genID: gen.id, bucketID: bucket.id, pairOffset: i)
                    )
                }
            }
        }

        for (chunkID, entries) in accum {
            let sorted = entries.sorted { $0.tokenOffset < $1.tokenOffset }
            ledger[chunkID] = sorted.map {
                .bucketRef(genID: $0.genID, bucketID: $0.bucketID, pairOffset: $0.pairOffset)
            }
        }
        self.dims = inferredDims
    }

    /// `.autoRecover` rehydration path.
    ///
    /// Four-pass algorithm:
    ///   1. Accumulate all data (with genID tagging), tracking which
    ///      generation(s) each chunkID appears in. Residual *values* are
    ///      not decoded here — only the cheap Requirement 13 integrity
    ///      check (center non-empty, indices decode, residuals blob size).
    ///   2. Resolve winner for each conflicting chunkID via pairwise
    ///      `(level DESC, created DESC, id DESC)` comparison.
    ///   3. Prune loser generation data from storage (atomically per gen).
    ///      Residual *values* are decoded here, and only here, for buckets
    ///      in a loser generation — re-encoding surviving pairs needs the
    ///      actual residual floats. This confines the one remaining
    ///      O(pairs×dims) cost to the rare corruption-recovery case.
    ///   4. Populate the ledger with `.bucketRef` tokens. `storage.replaceGeneration`
    ///      assigns fresh generation/bucket ids to survivors, so any
    ///      generation touched by the prune pass is re-walked post-prune to
    ///      pick up its new ids; untouched generations reuse the pairs
    ///      already decoded in step 1 at zero extra I/O.
    private func rehydrateAutoRecover(gens: [GenerationRecord]) async throws {
        // Local decoded bucket metadata — stashed during the accumulation
        // pass so the prune pass (step 3) can re-encode surviving entries
        // without extra storage I/O. Unlike the pre-#137 version, this does
        // NOT carry decoded residual values: those are only decoded in
        // step 3, and only for buckets in a loser generation.
        struct DecodedBucketMeta {
            let genID: Int64
            let record: BucketRecord
            let pairs: [IndexPair]
        }

        // Step 1 — Accumulation pass. Tracks which chunkIDs conflict and
        // stashes decoded (non-residual) bucket metadata; does not build
        // ledger entries directly since bucket ids for any generation
        // touched by the prune pass (step 3) become stale before step 4 runs.
        // chunkID → set of all generationIDs that claim it.
        var chunkGenSets: [Int64: Set<Int64>] = [:]
        var decodedBuckets: [DecodedBucketMeta] = []
        var inferredDims: Int? = nil

        for gen in gens {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            for bucket in buckets {
                let center = try Self.checkedBucketCenter(bucket, generationID: gen.id, inferredDims: &inferredDims)
                let d = center.count

                let pairs = try IndicesCodec.decode(bucket.indices)
                try Self.checkedResidualsSize(bucket, pairsCount: pairs.count, dims: d, generationID: gen.id)

                decodedBuckets.append(DecodedBucketMeta(genID: gen.id, record: bucket, pairs: pairs))

                for pair in pairs {
                    let chunkID = Int64(pair.chunkID)
                    chunkGenSets[chunkID, default: []].insert(gen.id)
                }
            }
        }

        guard let d = inferredDims else {
            // No buckets at all — nothing to do.
            return
        }

        // Step 2 — Winner resolution.
        // For each chunkID claimed by >1 generation, pick the winner via
        // pairwise (level DESC, created DESC, id DESC).
        let genLookup: [Int64: GenerationRecord] = Dictionary(uniqueKeysWithValues: gens.map { ($0.id, $0) })

        // winnerGenID[chunkID] = genID that wins for that chunkID.
        // Only populated for conflicting chunkIDs.
        var winnerGenID: [Int64: Int64] = [:]
        for (chunkID, genIDSet) in chunkGenSets where genIDSet.count > 1 {
            var winner: GenerationRecord? = nil
            for genID in genIDSet {
                guard let candidate = genLookup[genID] else { continue }
                guard let current = winner else {
                    winner = candidate
                    continue
                }
                // (level DESC, created DESC, id DESC)
                if candidate.level > current.level {
                    winner = candidate
                } else if candidate.level == current.level && candidate.created > current.created {
                    winner = candidate
                } else if candidate.level == current.level && candidate.created == current.created && candidate.id > current.id {
                    winner = candidate
                }
            }
            if let w = winner {
                winnerGenID[chunkID] = w.id
            }
        }

        // Step 3 — Storage prune.
        // Group conflicting chunkIDs by their loser generation(s).
        // loserChunkIDsByGen[loserGenID] = set of chunkIDs lost by that generation.
        var loserChunkIDsByGen: [Int64: Set<Int64>] = [:]
        for (chunkID, winnerID) in winnerGenID {
            guard let genIDSet = chunkGenSets[chunkID] else { continue }
            for loserGenID in genIDSet where loserGenID != winnerID {
                loserChunkIDsByGen[loserGenID, default: []].insert(chunkID)
            }
        }

        // Generations actually rewritten by `replaceGeneration` below (fresh
        // gen id + fresh bucket ids for survivors) — used by step 5 to know
        // which generations' `decodedBuckets` entries are now stale.
        var replacementGens: [GenerationRecord] = []

        // For each loser generation, compute surviving buckets and replace atomically.
        for (loserGenID, losingChunkIDs) in loserChunkIDsByGen {
            guard let loserGen = genLookup[loserGenID] else { continue }

            // Emit per-(chunkID, loser) warning log lines.
            let winnerForLog: [Int64: GenerationRecord] = Dictionary(
                uniqueKeysWithValues: losingChunkIDs.compactMap { chunkID -> (Int64, GenerationRecord)? in
                    guard let wid = winnerGenID[chunkID], let wgen = genLookup[wid] else { return nil }
                    return (chunkID, wgen)
                }
            )
            for chunkID in losingChunkIDs.sorted() {
                let winner = winnerForLog[chunkID]
                indexerLogger.warning("""
                    Recovered rehydrationConflict: chunkID=\(chunkID) \
                    winner=gen\(winner?.id ?? -1)(level:\(winner?.level ?? -1), created:\(winner?.created ?? Date.distantPast)) \
                    loser=gen\(loserGen.id)(level:\(loserGen.level), created:\(loserGen.created))
                    """)
            }

            // Build surviving buckets: filter out pairs whose chunkID is a
            // loser. Residual *values* are decoded here — and only here —
            // for buckets in this loser generation; re-encoding surviving
            // pairs needs the actual residual floats. This is the one
            // remaining O(pairs×dims) cost in this rehydration path,
            // confined to the rare corruption-recovery case.
            var survivingBuckets: [BucketRecord] = []
            var totalSurvivingEmbeddings = 0
            var survivingMinChunkID: Int64? = nil
            var survivingMaxChunkID: Int64? = nil

            for decoded in decodedBuckets where decoded.genID == loserGenID {
                let residuals = Q4Codec.decodeResiduals(decoded.record.residuals)
                // Filter pairs to those not in the losing set.
                var survivingPairs: [IndexPair] = []
                var survivingResidualValues: [Float] = []
                for (i, pair) in decoded.pairs.enumerated() {
                    let chunkID = Int64(pair.chunkID)
                    if losingChunkIDs.contains(chunkID) { continue }
                    survivingPairs.append(pair)
                    let base = i * d
                    for j in 0..<d {
                        survivingResidualValues.append(residuals[base + j])
                    }
                    survivingMinChunkID = survivingMinChunkID.map { min($0, chunkID) } ?? chunkID
                    survivingMaxChunkID = survivingMaxChunkID.map { max($0, chunkID) } ?? chunkID
                }
                if survivingPairs.isEmpty { continue }

                totalSurvivingEmbeddings += survivingPairs.count
                survivingBuckets.append(BucketRecord(
                    generationID: BucketRecord.unassigned,
                    center: decoded.record.center,
                    indices: IndicesCodec.encode(survivingPairs),
                    residuals: Q4Codec.encodeResiduals(survivingResidualValues)
                ))
            }

            // Build the surviving GenerationRecord with updated stats.
            let survivingRecord = GenerationRecord(
                level: loserGen.level,
                numEmbeddings: totalSurvivingEmbeddings,
                minChunkID: survivingMinChunkID ?? 0,
                maxChunkID: survivingMaxChunkID ?? 0,
                created: loserGen.created
            )

            // Atomically replace the loser in storage. May throw — propagates.
            // `replaceGeneration` assigns a fresh generation id (and fresh
            // bucket ids for every surviving bucket) — captured here so the
            // ledger-population pass (step 4 below) can re-walk exactly the
            // generations whose ids changed, instead of trusting the
            // now-stale ids in `decodedBuckets`.
            if let replaced = try await storage.replaceGeneration(
                losingGenerationID: loserGenID,
                survivingRecord: survivingRecord,
                survivingBuckets: survivingBuckets
            ) {
                replacementGens.append(replaced)
            }
        }

        // Step 3.5 — Correct stale numEmbeddings on all surviving (non-loser) generations.
        // A crash between step 9 (generation row insert) and step 10 (bucket inserts) in
        // performFlush() leaves the winner gen's numEmbeddings larger than the number of
        // pairs actually present in its buckets. The cascade walk in performFlush() uses
        // gen.numEmbeddings for `total` while allRows.count is derived from decoded bucket
        // pairs — if they diverge, ledgerOutOfSync fires. Correct here using
        // already-decoded bucket data (zero extra I/O). Only applies to generations
        // untouched by the prune pass — replacement generations built above already
        // carry a correct `numEmbeddings` (`totalSurvivingEmbeddings`).
        let loserGenIDs = Set(loserChunkIDsByGen.keys)
        for gen in gens where !loserGenIDs.contains(gen.id) {
            let actualCount = decodedBuckets
                .filter { $0.genID == gen.id }
                .reduce(0) { $0 + $1.pairs.count }
            if actualCount != gen.numEmbeddings {
                try await storage.updateGenerationEmbeddingCount(id: gen.id, count: actualCount)
            }
        }

        // Step 4 — Record recoveredConflictCount.
        recoveredConflictCount = winnerGenID.count

        // Step 5 — Ledger population with `.bucketRef` tokens.
        //
        // Untouched generations (not in `loserGenIDs`) keep their original
        // ids, so `decodedBuckets` entries for them are reused directly at
        // zero extra I/O. By construction, no chunkID within an untouched
        // generation can be on the losing side of a conflict — if it were,
        // this generation would itself be a loser for that chunkID and
        // would appear in `loserGenIDs`. So no winner-filtering is needed
        // for this branch.
        var populated: [Int64: [(tokenOffset: UInt32, genID: Int64, bucketID: Int64, pairOffset: Int)]] = [:]
        for decoded in decodedBuckets where !loserGenIDs.contains(decoded.genID) {
            for (i, pair) in decoded.pairs.enumerated() {
                let chunkID = Int64(pair.chunkID)
                populated[chunkID, default: []].append(
                    (tokenOffset: pair.tokenOffset, genID: decoded.genID, bucketID: decoded.record.id, pairOffset: i)
                )
            }
        }

        // Generations replaced by the prune pass: re-walk their (new)
        // buckets fresh from storage to pick up the post-prune ids.
        // Decodes only `indices` (O(pairs)), not residual values.
        for replacedGen in replacementGens {
            let buckets = try await storage.buckets(forGeneration: replacedGen.id)
            for bucket in buckets {
                let pairs = try IndicesCodec.decode(bucket.indices)
                for (i, pair) in pairs.enumerated() {
                    let chunkID = Int64(pair.chunkID)
                    populated[chunkID, default: []].append(
                        (tokenOffset: pair.tokenOffset, genID: replacedGen.id, bucketID: bucket.id, pairOffset: i)
                    )
                }
            }
        }

        for (chunkID, entries) in populated {
            let sorted = entries.sorted { $0.tokenOffset < $1.tokenOffset }
            ledger[chunkID] = sorted.map {
                .bucketRef(genID: $0.genID, bucketID: $0.bucketID, pairOffset: $0.pairOffset)
            }
        }
        self.dims = d
    }

    // MARK: - Public API

    /// Append per-token embeddings for one chunk to the in-memory L0
    /// buffer. Does not perform any I/O or trigger a cascade.
    ///
    /// `embeddings` is a row-major `m × dims` array; `m = embeddings.count
    /// / dims`. `dims` must be even (Q4Codec packs two nibbles per byte).
    /// `chunkID` must fit in `UInt32` so `(chunkID, tokenOffset)` pairs
    /// can be delta-encoded as in upstream Witchcraft (see ADR for
    /// bucket indices encoding).
    public func add(chunkID: Int64, embeddings: [Float], dims: Int) async throws {
        // Wait for any in-flight flush before mutating the ledger. The
        // leader's `performFlush()` body has multiple `await` points and
        // computes its row count from the ledger range; without this
        // gate, an `add` racing into a leader's suspension can add rows
        // to a chunkID that falls within the leader's captured
        // [pendingMin, pendingMax] window (chunk IDs reach
        // `indexer.add` out of order, since `SwitchcraftStore.add`
        // assigns the chunkID in `storage.upsertChunk` strictly before
        // it calls `indexer.add`, with an `await` in between). The
        // leader would then see m > expected and throw
        // `Error.ledgerOutOfSync`. Same waiter pattern as `flush()`.
        while flushInProgress {
            _ = try await withCheckedThrowingContinuation { (c: CheckedContinuation<CompactionEvent?, any Swift.Error>) in
                flushWaiters.append(c)
            }
        }

        precondition(dims > 0, "dims must be positive")
        precondition(dims % 2 == 0, "dims must be even (Q4Codec evenness precondition)")
        precondition(embeddings.count % dims == 0,
                     "embeddings.count (\(embeddings.count)) must be a multiple of dims (\(dims))")
        precondition(chunkID > 0, "chunkID must be positive")
        precondition(chunkID <= Int64(UInt32.max),
                     "chunkID (\(chunkID)) must fit in UInt32 (Witchcraft DocPtr parity)")
        if let existing = self.dims {
            precondition(existing == dims,
                         "indexer dims locked at \(existing); got \(dims)")
        } else {
            self.dims = dims
        }

        let m = embeddings.count / dims
        if m == 0 { return }

        var rows = [LedgerToken]()
        rows.reserveCapacity(m)
        for i in 0..<m {
            let start = i * dims
            rows.append(.materialized(Array(embeddings[start..<(start + dims)])))
        }
        ledger[chunkID, default: []].append(contentsOf: rows)

        pendingMinChunkID = pendingMinChunkID.map { min($0, chunkID) } ?? chunkID
        pendingMaxChunkID = pendingMaxChunkID.map { max($0, chunkID) } ?? chunkID
        pendingCount += m
    }

    /// Remove any ledger rows for the given chunkID.
    ///
    /// Called immediately before `add()` when recovering a partial orphan
    /// (a chunk rehydrated from storage with some existing bucket refs).
    /// Rehydration writes the partial rows into the ledger; calling this
    /// before `add()` clears them so the fresh `add()` can append a clean,
    /// complete set of rows rather than duplicating them.
    ///
    /// For full orphans (chunkID absent from the ledger after rehydration)
    /// this is a no-op. Includes the same flush-waiter gate as `add()` to
    /// avoid racing a concurrent `performFlush()` mid-decode.
    ///
    /// `pendingCount` is NOT modified: the removed rows are always retained
    /// or rehydrated rows (never pending), because the R2 pendingChunkIDs
    /// guard prevents reaching this method for any chunk that is currently
    /// pending. `pendingMin`/`pendingMax` are also left unchanged — a
    /// slightly wider range causes no incorrect behaviour.
    public func removeFromLedger(_ chunkID: Int64) async throws {
        while flushInProgress {
            _ = try await withCheckedThrowingContinuation { (c: CheckedContinuation<CompactionEvent?, any Swift.Error>) in
                flushWaiters.append(c)
            }
        }
        removedFromLedgerCount += ledger[chunkID]?.count ?? 0
        ledger.removeValue(forKey: chunkID)
    }

    /// Remove ledger rows for chunk ids that `SwitchcraftStore.vacuum()`
    /// just deleted from storage, and repoint every still-live chunk's
    /// `.bucketRef` tokens at their surviving pair's new offset in any
    /// bucket vacuum just compacted (issue #142).
    ///
    /// Unlike `removeFromLedger()` (used by the R1 orphan-recovery path to
    /// clear stale rows immediately before re-buffering fresh ones via
    /// `add()`), the `chunkIDs` deletion here is a **true delete**: the
    /// chunk ids are gone from storage for good, not about to be re-added.
    /// `removedFromLedgerCount` is deliberately **not** incremented here —
    /// that counter exists to compensate `performFlush()`'s cascade-walk
    /// total for rows that were cleared-and-about-to-be-refilled (see ADR
    /// 029 amendment, issue #132); vacuum already decremented the
    /// corresponding `generation.numEmbeddings` in storage for these exact
    /// chunk ids before calling this method, so the ledger and storage
    /// stay in lockstep with no compensation needed. Incrementing it here
    /// would double-subtract and desync a later flush.
    ///
    /// `remaps` come from `VacuumPlanBuilder.Result.bucketRefRemaps`,
    /// computed in the same per-bucket loop that produced the compacted
    /// bytes `applyVacuumPlan` already committed — this just applies the
    /// matching ledger-side update via the same
    /// `ledger[chunkID]?[tokenOffset] = .bucketRef(...)` pattern
    /// `performFlush()` already uses when it first materializes a
    /// bucket-ref (ADR 036). Applied as a blind overwrite, mirroring that
    /// precedent — `vacuum()`'s documented contract already excludes
    /// concurrent same-process `add()`/`index()` racing this call, so no
    /// defensive pre-check is needed.
    ///
    /// Includes the same flush-waiter gate as `add()`/`removeFromLedger()`
    /// to avoid racing a concurrent `performFlush()` mid-decode.
    public func applyVacuumLedgerUpdates(
        abandonedChunkIDs chunkIDs: Set<Int64>, remaps: [VacuumPlanBuilder.BucketRefRemap]
    ) async throws {
        while flushInProgress {
            _ = try await withCheckedThrowingContinuation { (c: CheckedContinuation<CompactionEvent?, any Swift.Error>) in
                flushWaiters.append(c)
            }
        }
        for chunkID in chunkIDs {
            ledger.removeValue(forKey: chunkID)
        }
        for remap in remaps {
            ledger[remap.chunkID]?[Int(remap.tokenOffset)] = .bucketRef(
                genID: remap.genID, bucketID: remap.bucketID, pairOffset: remap.newPairOffset
            )
        }
    }

    /// Drain any buffered embeddings into a new generation, cascading
    /// through higher levels as required by `IndexerConfig`. No-op when
    /// the buffer is empty.
    ///
    /// Concurrent callers serialise: the first caller becomes the leader
    /// and runs the flush body; subsequent callers that arrive while the
    /// leader is in-flight join a waiter list and resume when the leader
    /// returns. This is required because `performFlush()` has multiple
    /// `await` points (`storage.generations()`, `insertGeneration`, N ×
    /// `insertBucket`, M × `deleteGeneration`); without the guard, two
    /// concurrent callers can both pass the `pendingCount > 0` check and
    /// both write generations covering overlapping ledger rows. A later
    /// flush would then double-count those generations in `levelSums` and
    /// throw `Error.ledgerOutOfSync` even though the ledger is intact.
    ///
    /// **Error propagation policy**: if the leader throws, the same
    /// error is rethrown to *every* waiter (and to the leader's caller).
    /// All concurrent callers see one consistent outcome.
    ///
    /// **Fast path**: when `pendingCount == 0` and no flush is in flight,
    /// callers return immediately without taking the leader/waiter slow
    /// path. (When a flush is in flight, even pending=0 callers wait,
    /// because the in-flight flush is exactly the one their data needs.)
    @discardableResult
    public func flush() async throws -> CompactionEvent? {
        // Fast path: nothing to flush AND no in-flight flush.
        if pendingCount == 0 && !flushInProgress { return nil }

        // If a leader is already running, queue and wait for its result.
        // The leader rethrows its error to every waiter; on success
        // every waiter resumes with `nil` and returns.
        if flushInProgress {
            _ = try await withCheckedThrowingContinuation { (c: CheckedContinuation<CompactionEvent?, any Swift.Error>) in
                flushWaiters.append(c)
            }
            return nil
        }

        // Re-check the empty-pending guard now that we know we are the
        // leader candidate. The only way pendingCount can have changed
        // between the fast-path check and here is if we suspended — and
        // we have not. This guard preserves the original early-return
        // semantics for the no-pending case.
        guard pendingCount > 0, self.dims != nil,
              pendingMinChunkID != nil,
              pendingMaxChunkID != nil
        else {
            return nil
        }

        // Become the leader. Run the body; on completion (success or
        // throw), drain the waiter list with the same outcome.
        flushInProgress = true
        let event: CompactionEvent
        do {
            event = try await performFlush()
        } catch {
            let waiters = flushWaiters
            flushWaiters.removeAll()
            flushInProgress = false
            for w in waiters { w.resume(throwing: error) }
            throw error
        }
        // Write a fresh ledger snapshot now that the flush has committed
        // (issue #136 / ADR 034). We are still the leader (flushInProgress is
        // true), so no concurrent add()/removeFromLedger() can mutate the
        // ledger mid-encode. Best-effort: a snapshot write failure must NOT
        // fail the flush — the compaction already committed, and the snapshot
        // is a pure startup optimization backstopped by full rehydration.
        do {
            try await writeLedgerSnapshot()
        } catch {
            indexerLogger.warning("post-flush ledger snapshot write failed (\(String(describing: error), privacy: .public)); next startup will fall back to full rehydration")
        }
        let waiters = flushWaiters
        flushWaiters.removeAll()
        flushInProgress = false
        for w in waiters { w.resume(returning: nil) }
        return event
    }

    /// Persist the current in-memory ledger as an on-disk snapshot,
    /// unconditionally (issue #136 / ADR 034).
    ///
    /// Called by `SwitchcraftStore.shutdown()` after `flushAndClearPending()`
    /// and before `walCheckpoint()`. Unlike the post-flush write inside
    /// `flush()`, this runs even when nothing was pending — `flush()`'s
    /// `pendingCount == 0` fast path means an idle/read-only session never
    /// reaches `performFlush()`, so a snapshot loaded-and-cleared at startup
    /// would otherwise never be rewritten, silently regressing the next
    /// startup to a full walk.
    ///
    /// Becomes the flush leader for the duration so concurrent
    /// `add()`/`removeFromLedger()` calls (which gate on `flushInProgress`)
    /// cannot mutate the ledger mid-encode, mirroring `flush()`'s
    /// leader/waiter drain.
    ///
    /// Returns a wall-clock/size breakdown of the write (issue #144, REQ-1) —
    /// all-zero if `writeLedgerSnapshot()` took an early-return path. Callers
    /// that don't need the breakdown may discard it.
    @discardableResult
    public func persistSnapshot() async throws -> PersistSnapshotStats {
        while flushInProgress {
            _ = try await withCheckedThrowingContinuation { (c: CheckedContinuation<CompactionEvent?, any Swift.Error>) in
                flushWaiters.append(c)
            }
        }
        flushInProgress = true
        let stats: PersistSnapshotStats
        do {
            stats = try await writeLedgerSnapshot()
        } catch {
            let waiters = flushWaiters
            flushWaiters.removeAll()
            flushInProgress = false
            for w in waiters { w.resume(throwing: error) }
            throw error
        }
        let waiters = flushWaiters
        flushWaiters.removeAll()
        flushInProgress = false
        for w in waiters { w.resume(returning: nil) }
        return stats
    }

    /// Encode the current ledger + a fresh storage fingerprint into a
    /// `LedgerSnapshotRecord` and persist it (overwriting any prior snapshot).
    ///
    /// Must only be called while holding the flush leader gate
    /// (`flushInProgress == true`) so the ledger is stable across its `await`
    /// points. If `dims` has never been locked in (empty index), there is
    /// nothing meaningful to snapshot — any stale snapshot is cleared instead.
    @discardableResult
    private func writeLedgerSnapshot() async throws -> PersistSnapshotStats {
        guard let dims = self.dims else {
            try await storage.clearLedgerSnapshot()
            return .zero
        }
        let gens = try await storage.generations()
        let chunkCount = try await storage.chunkCount()
        let fingerprint = LedgerSnapshotFingerprint.compute(chunkCount: chunkCount, generations: gens)

        // Consistency guard (issue #136 / ADR 034). A snapshot is only valid at
        // points where the in-memory ledger is known-consistent with committed
        // storage. By the LSM invariant every ledger row belongs to exactly one
        // active generation, so the total ledger row count must equal
        // sum(gen.numEmbeddings). If they diverge, the ledger has been mutated
        // out-of-band relative to storage (e.g. storage rewritten directly
        // behind the Indexer's back, as when a crash leaves a partial-orphan
        // generation) — persisting a snapshot now would bake in a stale ledger
        // whose fingerprint falsely matches storage. Refuse to write, and clear
        // any prior snapshot so the next startup falls back to the full
        // rehydration walk (the crash-recovery safety net, unchanged).
        let ledgerRowCount = ledger.values.reduce(0) { $0 + $1.count }
        guard ledgerRowCount == fingerprint.totalEmbeddings else {
            indexerLogger.warning("Skipping ledger snapshot write: ledger rows (\(ledgerRowCount, privacy: .public)) != storage embeddings (\(fingerprint.totalEmbeddings, privacy: .public)); next startup will use full rehydration")
            try await storage.clearLedgerSnapshot()
            return .zero
        }

        let encodeStart = Date()
        let payload = LedgerSnapshotCodec.encode(ledger, dims: dims)
        let encodeSeconds = Date().timeIntervalSince(encodeStart)
        let record = LedgerSnapshotRecord(
            dims: dims,
            chunkCount: fingerprint.chunkCount,
            maxChunkID: fingerprint.maxChunkID,
            totalEmbeddings: fingerprint.totalEmbeddings,
            maxGenerationID: fingerprint.maxGenerationID,
            generationCount: fingerprint.generationCount,
            payload: payload
        )
        let writeStart = Date()
        try await storage.saveLedgerSnapshot(record)
        let writeSeconds = Date().timeIntervalSince(writeStart)

        return PersistSnapshotStats(
            encodeSeconds: encodeSeconds,
            writeSeconds: writeSeconds,
            bytesSerialized: payload.count,
            chunkCount: fingerprint.chunkCount
        )
    }

    /// One bucket's decoded contents, cached during a batched-materialization
    /// call so that a bucket referenced by multiple tokens is decoded once.
    private struct DecodedBucketData {
        let center: [Float]
        let pairs: [IndexPair]
        let residuals: [Float]
    }

    /// Per-call cache for batched bucket-ref materialization (issue #137
    /// Requirement 6): a bucket referenced by many tokens in the same call
    /// — e.g. several sqrt-sampled tokens landing in the same source bucket
    /// — is fetched and decoded at most once. A reference type so it can be
    /// threaded through a hot per-row loop (`collectAndMaterialize`) without
    /// `inout` plumbing. Not `Sendable`: always created and consumed within
    /// a single actor-isolated call, never shared across concurrent
    /// contexts or stored across calls.
    private final class BucketDecodeCache {
        var rawBucketsByGen: [Int64: [Int64: BucketRecord]] = [:]
        var decoded: [Int64: DecodedBucketData] = [:]
    }

    /// Fetch (if needed) and decode (if needed) the bucket `bucketID` in
    /// generation `genID`, consulting/populating `cache` so repeated calls
    /// for the same bucket within one `cache`'s lifetime do zero extra I/O
    /// or decode work.
    private func decodedBucket(
        genID: Int64, bucketID: Int64, chunkID: Int64, cache: BucketDecodeCache
    ) async throws -> DecodedBucketData {
        if let cached = cache.decoded[bucketID] {
            return cached
        }
        let record: BucketRecord
        if let existing = cache.rawBucketsByGen[genID]?[bucketID] {
            record = existing
        } else {
            let fetched: [BucketRecord]
            do {
                fetched = try await storage.buckets(forGeneration: genID)
            } catch {
                throw Error.bucketRefUnresolvable(chunkID: chunkID, bucketID: bucketID, cause: String(describing: error))
            }
            var byID: [Int64: BucketRecord] = [:]
            byID.reserveCapacity(fetched.count)
            for b in fetched { byID[b.id] = b }
            cache.rawBucketsByGen[genID] = byID
            guard let found = byID[bucketID] else {
                throw Error.bucketRefUnresolvable(
                    chunkID: chunkID, bucketID: bucketID,
                    cause: "bucket \(bucketID) not found in generation \(genID)"
                )
            }
            record = found
        }
        do {
            let center = Indexer.decodeFloat32LE(record.center)
            let pairs = try IndicesCodec.decode(record.indices)
            let residuals = Q4Codec.decodeResiduals(record.residuals)
            let result = DecodedBucketData(center: center, pairs: pairs, residuals: residuals)
            cache.decoded[bucketID] = result
            return result
        } catch {
            throw Error.bucketRefUnresolvable(chunkID: chunkID, bucketID: bucketID, cause: String(describing: error))
        }
    }

    /// Resolve one `.bucketRef` token to its `center + dequantized residual`
    /// row, via `decodedBucket` (so repeated refs into the same bucket incur
    /// no extra I/O/decode).
    ///
    /// - Throws: `Error.bucketRefUnresolvable` if the referenced generation's
    ///   buckets can't be fetched, the referenced bucketID isn't present, or
    ///   the bucket's decoded contents don't have room for `pairOffset` at
    ///   the expected `dims` — covering both storage-read failures and
    ///   residual corruption the cheap init-time integrity walk (Requirement
    ///   13) doesn't catch (it only checks blob sizes, not decoded values).
    private func resolveBucketRefRow(
        genID: Int64, bucketID: Int64, pairOffset: Int, chunkID: Int64, dims: Int, cache: BucketDecodeCache
    ) async throws -> [Float] {
        let decoded = try await decodedBucket(genID: genID, bucketID: bucketID, chunkID: chunkID, cache: cache)
        guard pairOffset >= 0, pairOffset < decoded.pairs.count,
              decoded.center.count == dims,
              (pairOffset + 1) * dims <= decoded.residuals.count
        else {
            throw Error.bucketRefUnresolvable(
                chunkID: chunkID, bucketID: bucketID,
                cause: "pairOffset \(pairOffset) out of range or bucket size mismatch (dims=\(dims))"
            )
        }
        let base = pairOffset * dims
        var row = [Float](repeating: 0, count: dims)
        for j in 0..<dims {
            row[j] = decoded.center[j] + decoded.residuals[base + j]
        }
        return row
    }

    /// Resolve a list of ledger tokens into flat row data, in the same
    /// order as `tokens`. `.materialized` tokens are copied directly with
    /// no I/O; `.bucketRef` tokens are resolved via `resolveBucketRefRow`,
    /// batching bucket decode by bucketID (issue #137 Requirement 6). Used
    /// by `ledgerContents()` (test-only accessor); the `performFlush()` hot
    /// path uses `collectAndMaterialize` instead, which inlines this same
    /// per-token resolution directly into its collection loop to avoid the
    /// extra intermediate `(chunkID, token)` array this general-purpose
    /// entry point allocates.
    private func materialize(
        _ tokens: [(chunkID: Int64, token: LedgerToken)],
        dims: Int
    ) async throws -> [Float] {
        let cache = BucketDecodeCache()
        var out = [Float]()
        out.reserveCapacity(tokens.count * dims)
        for (chunkID, token) in tokens {
            switch token {
            case .materialized(let row):
                out.append(contentsOf: row)
            case .bucketRef(let genID, let bucketID, let pairOffset):
                out.append(contentsOf: try await resolveBucketRefRow(
                    genID: genID, bucketID: bucketID, pairOffset: pairOffset, chunkID: chunkID, dims: dims, cache: cache
                ))
            }
        }
        return out
    }

    /// Collect ledger rows for chunk ids in `[min, max]`, in (chunkID
    /// ascending, tokenOffset ascending) order, materializing each row
    /// in-place as it's collected. Returns the sorted chunk ids, the flat
    /// materialized values, parallel per-row `(chunkID, tokenOffset)`
    /// arrays, and a `chunkID -> row start index` map — the latter lets the
    /// sqrt-sample training step (step 4 of `performFlush`) slice sampled
    /// rows directly out of the flat buffer instead of decoding a second
    /// time.
    ///
    /// This is a single pass (no intermediate token list) so the common
    /// case — a chunk range with no bucket-refs yet, e.g. the first flush
    /// of a fresh corpus — costs the same as the pre-#137 direct-copy loop.
    private func collectAndMaterialize(
        min: Int64, max: Int64, expectedTotal: Int, dims: Int
    ) async throws -> (
        sortedChunkIDs: [Int64],
        flat: [Float],
        rowChunkIDs: [Int64],
        rowTokenOffsets: [UInt32],
        chunkRowStart: [Int64: Int]
    ) {
        let sorted = ledger.keys
            .filter { $0 >= min && $0 <= max }
            .sorted()
        var flat: [Float] = []
        flat.reserveCapacity(expectedTotal * dims)
        var rowChunkIDs: [Int64] = []
        rowChunkIDs.reserveCapacity(expectedTotal)
        var rowTokenOffsets: [UInt32] = []
        rowTokenOffsets.reserveCapacity(expectedTotal)
        var chunkRowStart: [Int64: Int] = [:]
        chunkRowStart.reserveCapacity(sorted.count)
        let cache = BucketDecodeCache()

        for chunkID in sorted {
            let tokens = ledger[chunkID]!
            chunkRowStart[chunkID] = rowChunkIDs.count
            for (idx, token) in tokens.enumerated() {
                switch token {
                case .materialized(let row):
                    flat.append(contentsOf: row)
                case .bucketRef(let genID, let bucketID, let pairOffset):
                    flat.append(contentsOf: try await resolveBucketRefRow(
                        genID: genID, bucketID: bucketID, pairOffset: pairOffset, chunkID: chunkID, dims: dims, cache: cache
                    ))
                }
                rowChunkIDs.append(chunkID)
                rowTokenOffsets.append(UInt32(idx))
            }
        }
        return (sorted, flat, rowChunkIDs, rowTokenOffsets, chunkRowStart)
    }

    /// Body of the flush operation. Must only be invoked by `flush()`,
    /// which holds the `flushInProgress` leader guard. Concurrent calls
    /// to this method would re-introduce the race documented on `flush()`.
    private func performFlush() async throws -> CompactionEvent {
        guard pendingCount > 0, let dims = self.dims,
              let pendingMin = pendingMinChunkID,
              let pendingMax = pendingMaxChunkID
        else {
            // flush() always guards these preconditions before calling performFlush().
            fatalError("performFlush() called without pending data — invariant violated in flush()")
        }
        let pending = pendingCount
        // Capture the count of rehydrated rows removed by removeFromLedger()
        // since the last flush. The cascade walk subtracts these from `pending`
        // so that the partial-orphan R1 path (which clears stale rehydrated
        // rows and re-buffers them) doesn't cause `total` to overcount and
        // produce a false ledgerOutOfSync (ADR 029 amendment, issue #132).
        let capturedRemovedCount = removedFromLedgerCount

        let allGens = try await storage.generations()

        // 1. Cascade walk: pick the lowest level whose capacity holds
        // pending + the sum of embeddings already at that level (after
        // accumulating the lower levels merged in along the way).
        var total = pending - capturedRemovedCount
        var targetLevel = 0
        let levelSums: [Int: Int] = Dictionary(grouping: allGens, by: { $0.level })
            .mapValues { $0.reduce(0) { $0 + $1.numEmbeddings } }
        while true {
            let cap = config.levelCapacity(targetLevel)
            total += levelSums[targetLevel] ?? 0
            if total <= cap { break }
            targetLevel += 1
            // Defensive cap — in practice cascade beyond level ~10 is
            // many billions of embeddings; >32 indicates a config bug.
            precondition(targetLevel <= 32,
                         "Indexer cascade exceeded 32 levels — config likely broken")
        }

        // 2. Determine the chunk-ID range covered by the new generation.
        // Witchcraft `index_chunks` derives this from `min(min_chunk_rowid)
        // over generations at level <= target_level` and `max(rowid) over
        // chunk` — equivalent to "pending range ∪ ranges of merged gens".
        var minChunkID = pendingMin
        var maxChunkID = pendingMax
        var mergedGens = allGens.filter { $0.level <= targetLevel }
        for g in mergedGens {
            minChunkID = min(minChunkID, g.minChunkID)
            maxChunkID = max(maxChunkID, g.maxChunkID)
        }

        let inputSegments = mergedGens.count + 1
        let inputBytes = total * dims * 4
        let trigger = targetLevel > 0 ? "cascade" : "manual"
        let compactStart = Date()
        indexerLogger.info("[Compact] started: input_segments=\(inputSegments, privacy: .public) input_bytes=\(inputBytes, privacy: .public) trigger=\(trigger, privacy: .public)")
        do {

        // 3. Collect every embedding in [minChunkID, maxChunkID] from the
        // in-memory ledger, in (chunkID ascending, tokenOffset ascending)
        // order. `collectAndMaterialize` batches bucket-ref decode by
        // bucketID (issue #137 Requirement 6) and builds `allFlat` directly
        // — no intermediate `[[Float]]` — reused as-is in step 7 (no second
        // copy) and sliced in step 4 for the sqrt-sample training set (no
        // second bucket decode there either).
        var (sortedChunkIDs, allFlat, rowChunkIDs, rowTokenOffsets, chunkRowStart) =
            try await collectAndMaterialize(min: minChunkID, max: maxChunkID, expectedTotal: total, dims: dims)
        var m = rowChunkIDs.count
        if m != total {
            // Mid-operation ledger–storage divergence: the cascade walk's
            // chunk-ID range swept in rows from higher-level gens not included
            // in mergedGens (ADR 030). This happens when an add() workload
            // re-buffers embeddings for chunkIDs already rehydrated from a
            // higher-level gen, making the ledger count exceed what the storage
            // generation counters expected.
            //
            // Self-recovery: identify surprise gens (at levels above targetLevel
            // whose chunk-ID ranges overlap [minChunkID..maxChunkID]), absorb
            // them into mergedGens, extend the range, and re-collect the ledger.
            let mergedGenIDs = Set(mergedGens.map { $0.id })
            let surpriseGens = allGens.filter { g in
                !mergedGenIDs.contains(g.id) &&
                g.minChunkID <= maxChunkID &&
                g.maxChunkID >= minChunkID
            }
            guard !surpriseGens.isEmpty else {
                throw Error.ledgerOutOfSync(ledgerRows: m, expected: total)
            }
            indexerLogger.warning("[Compact] mid-operation ledger divergence detected (ledger=\(m, privacy: .public) expected=\(total, privacy: .public)); absorbing \(surpriseGens.count, privacy: .public) surprise gen(s) — see ADR 030")
            mergedGens.append(contentsOf: surpriseGens)
            for g in surpriseGens {
                minChunkID = min(minChunkID, g.minChunkID)
                maxChunkID = max(maxChunkID, g.maxChunkID)
                total += g.numEmbeddings
                targetLevel = max(targetLevel, g.level)
            }
            (sortedChunkIDs, allFlat, rowChunkIDs, rowTokenOffsets, chunkRowStart) =
                try await collectAndMaterialize(min: minChunkID, max: maxChunkID, expectedTotal: total, dims: dims)
            m = rowChunkIDs.count
            if m != total {
                throw Error.ledgerOutOfSync(ledgerRows: m, expected: total)
            }
            indexerLogger.info("[Compact] self-recovered: extended to \(mergedGens.count, privacy: .public) merged gens, targetLevel=\(targetLevel, privacy: .public), total=\(total, privacy: .public)")
        }

        // Compute final post-recovery values for the CompactionEvent (ADR 030:
        // self-recovery may have appended to mergedGens and incremented total).
        let finalInputSegments = mergedGens.count + 1
        let finalInputBytes = UInt64(total * dims * 4)
        let finalTrigger: CompactionEvent.Trigger = targetLevel > 0 ? .cascade : .manual

        // 4. Build the per-chunk sqrt-sample training set, matching
        // Witchcraft's `sample_embeddings_for_kmeans`. Sampled rows are
        // sliced directly out of `allFlat` (already materialized in step 3)
        // via `chunkRowStart` — no second bucket decode is needed, so this
        // consumer is "free" once step 3's batched materialization has run.
        var rng = SplitMix64(seed: config.seed)
        var trainFlat: [Float] = []
        trainFlat.reserveCapacity(m * dims)
        var trainM = 0
        for chunkID in sortedChunkIDs {
            let chunkM = ledger[chunkID]!.count
            let sampleK = Int(Double(chunkM).squareRoot().rounded(.up))
            let picked = Self.sampleDistinct(
                count: min(sampleK, chunkM), from: chunkM, rng: &rng
            )
            let base = chunkRowStart[chunkID]!
            for i in picked {
                let rowStart = (base + i) * dims
                trainFlat.append(contentsOf: allFlat[rowStart..<(rowStart + dims)])
                trainM += 1
            }
        }

        // 5. Compute k via Witchcraft's formula, clamped to the training
        // set size and to >= 1.
        var k = Int((config.kCoefficient * Double(m).squareRoot()).rounded())
        k = max(k, 1)
        if trainM < k {
            k = max(trainM / 4, 1)
        }
        k = min(k, trainM)

        // 6. Run k-means on the training set.
        let kmeansResult = KMeans.cluster(
            data: trainFlat, dims: dims, clusters: k,
            maxIterations: config.kmeansIterations,
            rng: &rng
        )
        let centroids = kmeansResult.centroids

        // 7. Assign every row in [minChunkID, maxChunkID] to its nearest
        // centroid using the trained centers. allFlat was built in step 3
        // — no second copy is needed.
        let assignments = KMeans.assign(
            data: allFlat, dims: dims, centroids: centroids
        )

        // 8. Bucketise row indices.
        var bucketMembers: [[Int]] = Array(repeating: [], count: k)
        for (rowIdx, cluster) in assignments.enumerated() {
            bucketMembers[cluster].append(rowIdx)
        }

        // 9. Insert the new generation. We do this before bucket inserts
        // so we have a stable generationID to thread through.
        let genRecord = GenerationRecord(
            level: targetLevel,
            numEmbeddings: m,
            minChunkID: minChunkID,
            maxChunkID: maxChunkID,
            created: Date()
        )
        try Task.checkCancellation()
        let insertedGen = try await storage.insertGeneration(genRecord)

        // 10. Write one bucket per centroid (k buckets total, including
        // any with zero assignments — the spec requires bucket count == k).
        for c in 0..<k {
            let centerStart = c * dims
            let centerSlice = Array(centroids[centerStart..<(centerStart + dims)])

            // Sort cluster members by (chunkID, tokenOffset) ascending so
            // the encoded indices satisfy the Search pipeline invariant.
            let members = bucketMembers[c].sorted { a, b in
                if rowChunkIDs[a] != rowChunkIDs[b] {
                    return rowChunkIDs[a] < rowChunkIDs[b]
                }
                return rowTokenOffsets[a] < rowTokenOffsets[b]
            }

            var pairs = [IndexPair]()
            pairs.reserveCapacity(members.count)
            var residualValues = [Float]()
            residualValues.reserveCapacity(members.count * dims)
            for rowIdx in members {
                pairs.append(IndexPair(
                    chunkID: UInt32(rowChunkIDs[rowIdx]),
                    tokenOffset: rowTokenOffsets[rowIdx]
                ))
                let rowStart = rowIdx * dims
                for j in 0..<dims {
                    residualValues.append(allFlat[rowStart + j] - centerSlice[j])
                }
            }

            // autoreleasepool drains transient Foundation Data objects produced
            // by encodeFloat32LE, IndicesCodec.encode, and Q4Codec.encodeResiduals
            // after each bucket iteration. The await is outside the pool so the
            // pool closure remains synchronous (no await inside autoreleasepool).
            let bucket = autoreleasepool {
                BucketRecord(
                    generationID: insertedGen.id,
                    center: Self.encodeFloat32LE(centerSlice),
                    indices: IndicesCodec.encode(pairs),
                    residuals: Q4Codec.encodeResiduals(residualValues)
                )
            }
            let insertedBucket = try await storage.insertBucket(bucket)

            // Requirement 3 (issue #137): transition the tokens just
            // flushed from materialized to bucket-ref form, pointing at
            // the generation/bucket/pairOffset just written. `pairOffset`
            // is the token's index within `pairs` (== its index within
            // `members`, since `pairs` was built by iterating `members` in
            // this exact order), matching what
            // `IndicesCodec.decode(bucket.indices)` will later return.
            // Chunks outside this flush's [minChunkID, maxChunkID] range
            // are untouched. Safe against concurrent mutation: `add()` /
            // `removeFromLedger()` gate on `flushInProgress`, which is
            // still held by the caller (`flush()`) for the duration of
            // `performFlush()`.
            for (pairOffset, rowIdx) in members.enumerated() {
                let chunkID = rowChunkIDs[rowIdx]
                let tokenOffset = Int(rowTokenOffsets[rowIdx])
                ledger[chunkID]?[tokenOffset] = .bucketRef(
                    genID: insertedGen.id, bucketID: insertedBucket.id, pairOffset: pairOffset
                )
            }
        }

        // 11. Delete merged-in generations now that the new one is
        // persisted. Storage backends cascade FK deletes to buckets.
        for g in mergedGens {
            try await storage.deleteGeneration(id: g.id)
        }

        // 12. Reset pending tracking. The ledger stays — cascades higher
        // up the LSM tree may re-cluster these rows again.
        pendingMinChunkID = nil
        pendingMaxChunkID = nil
        pendingCount = 0
        removedFromLedgerCount = 0
        let outputBytes = m * dims * 4
        indexerLogger.info("[Compact] finished: input_segments=\(finalInputSegments, privacy: .public) output_segments=\(1, privacy: .public) input_bytes=\(finalInputBytes, privacy: .public) output_bytes=\(outputBytes, privacy: .public) elapsed_ms=\(Int(Date().timeIntervalSince(compactStart) * 1000), privacy: .public)")
        return CompactionEvent(
            inputBytes: finalInputBytes,
            trigger: finalTrigger,
            inputSegments: finalInputSegments,
            startedAt: compactStart
        )
        } catch is CancellationError {
            let elapsedMs = Int(Date().timeIntervalSince(compactStart) * 1000)
            indexerLogger.info("[Compact] cancelled: processed_segments=\(0, privacy: .public)/\(inputSegments, privacy: .public) elapsed_ms=\(elapsedMs, privacy: .public)")
            throw CancellationError()
        } catch {
            let elapsedMs = Int(Date().timeIntervalSince(compactStart) * 1000)
            indexerLogger.info("[Compact] failed: input_segments=\(inputSegments, privacy: .public) elapsed_ms=\(elapsedMs, privacy: .public) error=\(error.localizedDescription, privacy: .auto)")
            throw error
        }
    }

    /// Wipe every persisted generation (and its buckets, via FK
    /// cascade) and reset the in-memory L0 buffer. Documents and chunks
    /// are not touched.
    public func clearIndex() async throws {
        let gens = try await storage.generations()
        for g in gens {
            try await storage.deleteGeneration(id: g.id)
        }
        // Drop any persisted ledger snapshot too (issue #136): a stale one is
        // caught safely by the fingerprint check on next init, but leaving a
        // potentially large orphaned blob on disk is poor hygiene.
        try await storage.clearLedgerSnapshot()
        ledger.removeAll()
        dims = nil
        pendingMinChunkID = nil
        pendingMaxChunkID = nil
        pendingCount = 0
        removedFromLedgerCount = 0
    }

    // MARK: - Testing helpers

    /// Returns a fully-materialized copy of the current ledger (chunkID →
    /// per-token embeddings), resolving any `.bucketRef` tokens via a single
    /// batched `materialize` call (issue #137 Requirement 7). Internal, not
    /// part of the public API. Accessible from `@testable import
    /// SwitchcraftCore`. Renamed from `ledgerSnapshot` (issue #136) to
    /// disambiguate from the persisted `LedgerSnapshotRecord` fast-path
    /// feature — this is an in-memory copy for test assertions, not the
    /// on-disk snapshot.
    ///
    /// **Breaking change** (issue #137): now `async throws` since resolving
    /// bucket-refs may read from storage. All in-package call sites are
    /// test-only and already run in an async test context.
    func ledgerContents() async throws -> [Int64: [[Float]]] {
        guard let dims = self.dims else { return [:] }
        var queue: [(chunkID: Int64, token: LedgerToken)] = []
        var order: [(chunkID: Int64, count: Int)] = []
        order.reserveCapacity(ledger.count)
        for (chunkID, tokens) in ledger {
            order.append((chunkID: chunkID, count: tokens.count))
            for token in tokens {
                queue.append((chunkID: chunkID, token: token))
            }
        }
        let flat = try await materialize(queue, dims: dims)
        var out: [Int64: [[Float]]] = [:]
        out.reserveCapacity(order.count)
        var offset = 0
        for (chunkID, count) in order {
            var rows: [[Float]] = []
            rows.reserveCapacity(count)
            for _ in 0..<count {
                let start = offset * dims
                rows.append(Array(flat[start..<(start + dims)]))
                offset += 1
            }
            out[chunkID] = rows
        }
        return out
    }

    /// True when `init` populated the ledger from a persisted snapshot
    /// (fast path) rather than running the full rehydration walk. Diagnostic
    /// signal for the snapshot fast-path reproducer/fallback tests and for
    /// consumers that want to observe which startup path was taken (issue
    /// #136). Public so it is reachable across module boundaries (e.g. from
    /// `SwitchcraftStore` in the `Switchcraft` module).
    public private(set) var didUseSnapshotFastPath: Bool = false

    // MARK: - Helpers

    /// Pack `[Float]` (Float32) as little-endian bytes. Encoding by hand
    /// (rather than `withUnsafeBytes` raw memory copy) makes the on-disk
    /// format independent of host endianness.
    static func encodeFloat32LE(_ values: [Float]) -> Data {
        var out = Data(capacity: values.count * 4)
        for v in values {
            let bits = v.bitPattern
            out.append(UInt8(bits & 0xFF))
            out.append(UInt8((bits >> 8) & 0xFF))
            out.append(UInt8((bits >> 16) & 0xFF))
            out.append(UInt8((bits >> 24) & 0xFF))
        }
        return out
    }

    /// Decode `[Float]` from little-endian Float32 bytes.
    static func decodeFloat32LE(_ data: Data) -> [Float] {
        precondition(data.count % 4 == 0,
                     "Float32 LE buffer length must be a multiple of 4")
        var out = [Float]()
        out.reserveCapacity(data.count / 4)
        let base = data.startIndex
        for i in stride(from: 0, to: data.count, by: 4) {
            let b0 = UInt32(data[base + i])
            let b1 = UInt32(data[base + i + 1]) << 8
            let b2 = UInt32(data[base + i + 2]) << 16
            let b3 = UInt32(data[base + i + 3]) << 24
            out.append(Float(bitPattern: b0 | b1 | b2 | b3))
        }
        return out
    }

    /// Floyd's algorithm for sampling `count` distinct integers from
    /// `[0, population)`. Mirrors `KMeans.sampleDistinct` byte-for-byte
    /// (including RNG consumption when `count == population`) so two
    /// builds with the same seed produce identical training-set selections.
    private static func sampleDistinct<R: RandomNumberGenerator>(
        count: Int, from population: Int, rng: inout R
    ) -> [Int] {
        precondition(count <= population)
        var picked = Set<Int>()
        picked.reserveCapacity(count)
        var ordered = [Int]()
        ordered.reserveCapacity(count)
        for j in (population - count)..<population {
            let t = Int(rng.next() % UInt64(j + 1))
            if picked.insert(t).inserted {
                ordered.append(t)
            } else {
                picked.insert(j)
                ordered.append(j)
            }
        }
        return ordered
    }
}
