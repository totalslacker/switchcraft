// SPDX-License-Identifier: Apache-2.0
import Foundation
import os

private let indexerLogger = Logger(subsystem: "com.switchcraft.core", category: "Indexer")

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
    public enum Error: Swift.Error, Sendable, Equatable {
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
        /// embedding dimension (which must be fixed per database).
        case rehydrationBucketCorrupt(generationID: Int64)
    }

    // MARK: - State

    private let storage: any SwitchcraftStorage
    public let config: IndexerConfig

    /// Every embedding ever added, keyed by chunkID. Token order within
    /// each chunk is the order of the embeddings passed to `add`.
    private var ledger: [Int64: [[Float]]] = [:]

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
    private var flushWaiters: [CheckedContinuation<Void, any Swift.Error>] = []

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

        switch config.rehydrationConflictBehavior {
        case .throwError:
            try await rehydrateThrowError(gens: gens)
        case .autoRecover:
            try await rehydrateAutoRecover(gens: gens)
        }
    }

    /// `.throwError` rehydration path — unchanged from the original implementation.
    /// Throws `rehydrationConflict` on the first detected chunkID overlap.
    private func rehydrateThrowError(gens: [GenerationRecord]) async throws {
        // Accumulate (tokenOffset, embedding) triples per chunkID across all
        // buckets of all generations before populating the ledger. Tokens for
        // a single chunkID may be spread across multiple buckets, so the sort
        // step below is required to restore the original tokenOffset order that
        // performFlush() assumes (row index == tokenOffset).
        var accum: [Int64: [(tokenOffset: UInt32, embedding: [Float])]] = [:]
        // Maps chunkID → the generationID where it was first seen. Used to
        // detect the storage-corruption case of a chunkID in two active gens.
        var chunkGenMap: [Int64: Int64] = [:]
        var inferredDims: Int? = nil

        for gen in gens {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            for bucket in buckets {
                let center = Indexer.decodeFloat32LE(bucket.center)
                guard !center.isEmpty else {
                    throw Error.rehydrationBucketCorrupt(generationID: gen.id)
                }
                let d = center.count
                if let existing = inferredDims, existing != d {
                    throw Error.rehydrationBucketCorrupt(generationID: gen.id)
                }
                inferredDims = d

                let pairs = try IndicesCodec.decode(bucket.indices)
                let residuals = Q4Codec.decodeResiduals(bucket.residuals)
                guard residuals.count == pairs.count * d else {
                    throw Error.rehydrationBucketCorrupt(generationID: gen.id)
                }

                for (i, pair) in pairs.enumerated() {
                    let chunkID = Int64(pair.chunkID)
                    if let seenInGen = chunkGenMap[chunkID], seenInGen != gen.id {
                        throw Error.rehydrationConflict(chunkID: chunkID)
                    }
                    chunkGenMap[chunkID] = gen.id

                    var embedding = [Float]()
                    embedding.reserveCapacity(d)
                    let base = i * d
                    for j in 0..<d {
                        embedding.append(center[j] + residuals[base + j])
                    }
                    accum[chunkID, default: []].append(
                        (tokenOffset: pair.tokenOffset, embedding: embedding)
                    )
                }
            }
        }

        for (chunkID, entries) in accum {
            let sorted = entries.sorted { $0.tokenOffset < $1.tokenOffset }
            ledger[chunkID] = sorted.map { $0.embedding }
        }
        self.dims = inferredDims
    }

    /// `.autoRecover` rehydration path.
    ///
    /// Four-pass algorithm:
    ///   1. Accumulate all data (with genID tagging), tracking which
    ///      generation(s) each chunkID appears in.
    ///   2. Resolve winner for each conflicting chunkID via pairwise
    ///      `(level DESC, created DESC, id DESC)` comparison.
    ///   3. Prune loser generation data from storage (atomically per gen).
    ///   4. Populate the ledger, filtering conflicted chunkIDs to their winner.
    private func rehydrateAutoRecover(gens: [GenerationRecord]) async throws {
        // Local decoded bucket — stashed during the accumulation pass so the
        // prune pass can re-encode surviving entries without extra storage I/O.
        struct DecodedBucket {
            let genID: Int64
            let record: BucketRecord
            let center: [Float]
            let pairs: [IndexPair]
            let residuals: [Float]
        }

        // Step 1 — Accumulation pass.
        // genID-tagged accum: chunkID → [(genID, tokenOffset, embedding)]
        var accum: [Int64: [(genID: Int64, tokenOffset: UInt32, embedding: [Float])]] = [:]
        // chunkID → set of all generationIDs that claim it.
        var chunkGenSets: [Int64: Set<Int64>] = [:]
        var decodedBuckets: [DecodedBucket] = []
        var inferredDims: Int? = nil

        for gen in gens {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            for bucket in buckets {
                let center = Indexer.decodeFloat32LE(bucket.center)
                guard !center.isEmpty else {
                    throw Error.rehydrationBucketCorrupt(generationID: gen.id)
                }
                let d = center.count
                if let existing = inferredDims, existing != d {
                    throw Error.rehydrationBucketCorrupt(generationID: gen.id)
                }
                inferredDims = d

                let pairs = try IndicesCodec.decode(bucket.indices)
                let residuals = Q4Codec.decodeResiduals(bucket.residuals)
                guard residuals.count == pairs.count * d else {
                    throw Error.rehydrationBucketCorrupt(generationID: gen.id)
                }

                decodedBuckets.append(DecodedBucket(
                    genID: gen.id,
                    record: bucket,
                    center: center,
                    pairs: pairs,
                    residuals: residuals
                ))

                for (i, pair) in pairs.enumerated() {
                    let chunkID = Int64(pair.chunkID)
                    chunkGenSets[chunkID, default: []].insert(gen.id)

                    var embedding = [Float]()
                    embedding.reserveCapacity(d)
                    let base = i * d
                    for j in 0..<d {
                        embedding.append(center[j] + residuals[base + j])
                    }
                    accum[chunkID, default: []].append(
                        (genID: gen.id, tokenOffset: pair.tokenOffset, embedding: embedding)
                    )
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

            // Build surviving buckets: filter out pairs whose chunkID is a loser.
            var survivingBuckets: [BucketRecord] = []
            var totalSurvivingEmbeddings = 0
            var survivingMinChunkID: Int64? = nil
            var survivingMaxChunkID: Int64? = nil

            for decoded in decodedBuckets where decoded.genID == loserGenID {
                // Filter pairs to those not in the losing set.
                var survivingPairs: [IndexPair] = []
                var survivingResidualValues: [Float] = []
                for (i, pair) in decoded.pairs.enumerated() {
                    let chunkID = Int64(pair.chunkID)
                    if losingChunkIDs.contains(chunkID) { continue }
                    survivingPairs.append(pair)
                    let base = i * d
                    for j in 0..<d {
                        survivingResidualValues.append(decoded.residuals[base + j])
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
            _ = try await storage.replaceGeneration(
                losingGenerationID: loserGenID,
                survivingRecord: survivingRecord,
                survivingBuckets: survivingBuckets
            )
        }

        // Step 4 — Record recoveredConflictCount.
        recoveredConflictCount = winnerGenID.count

        // Step 5 — Ledger population. For conflicting chunkIDs, only use
        // entries from the winning generation.
        for (chunkID, entries) in accum {
            let filteredEntries: [(tokenOffset: UInt32, embedding: [Float])]
            if let winnerID = winnerGenID[chunkID] {
                filteredEntries = entries.compactMap { e in
                    e.genID == winnerID ? (tokenOffset: e.tokenOffset, embedding: e.embedding) : nil
                }
            } else {
                filteredEntries = entries.map { (tokenOffset: $0.tokenOffset, embedding: $0.embedding) }
            }
            let sorted = filteredEntries.sorted { $0.tokenOffset < $1.tokenOffset }
            ledger[chunkID] = sorted.map { $0.embedding }
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
            try await withCheckedThrowingContinuation { c in
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

        var rows = [[Float]]()
        rows.reserveCapacity(m)
        for i in 0..<m {
            let start = i * dims
            rows.append(Array(embeddings[start..<(start + dims)]))
        }
        ledger[chunkID, default: []].append(contentsOf: rows)

        pendingMinChunkID = pendingMinChunkID.map { min($0, chunkID) } ?? chunkID
        pendingMaxChunkID = pendingMaxChunkID.map { max($0, chunkID) } ?? chunkID
        pendingCount += m
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
    public func flush() async throws {
        // Fast path: nothing to flush AND no in-flight flush.
        if pendingCount == 0 && !flushInProgress { return }

        // If a leader is already running, queue and wait for its result.
        // The leader rethrows its error to every waiter; on success
        // every waiter resumes with `()` and returns.
        if flushInProgress {
            try await withCheckedThrowingContinuation { c in
                flushWaiters.append(c)
            }
            return
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
            return
        }

        // Become the leader. Run the body; on completion (success or
        // throw), drain the waiter list with the same outcome.
        flushInProgress = true
        do {
            try await performFlush()
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
        for w in waiters { w.resume(returning: ()) }
    }

    /// Body of the flush operation. Must only be invoked by `flush()`,
    /// which holds the `flushInProgress` leader guard. Concurrent calls
    /// to this method would re-introduce the race documented on `flush()`.
    private func performFlush() async throws {
        guard pendingCount > 0, let dims = self.dims,
              let pendingMin = pendingMinChunkID,
              let pendingMax = pendingMaxChunkID
        else {
            return
        }
        let pending = pendingCount

        let allGens = try await storage.generations()

        // 1. Cascade walk: pick the lowest level whose capacity holds
        // pending + the sum of embeddings already at that level (after
        // accumulating the lower levels merged in along the way).
        var total = pending
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
        let mergedGens = allGens.filter { $0.level <= targetLevel }
        for g in mergedGens {
            minChunkID = min(minChunkID, g.minChunkID)
            maxChunkID = max(maxChunkID, g.maxChunkID)
        }

        // 3. Collect every embedding in [minChunkID, maxChunkID] from
        // the in-memory ledger, in (chunkID ascending, tokenOffset
        // ascending) order.
        var allRows: [[Float]] = []
        var rowChunkIDs: [Int64] = []
        var rowTokenOffsets: [UInt32] = []
        let sortedChunkIDs = ledger.keys
            .filter { $0 >= minChunkID && $0 <= maxChunkID }
            .sorted()
        for chunkID in sortedChunkIDs {
            let tokens = ledger[chunkID]!
            for (idx, row) in tokens.enumerated() {
                allRows.append(row)
                rowChunkIDs.append(chunkID)
                rowTokenOffsets.append(UInt32(idx))
            }
        }
        let m = allRows.count
        if m != total {
            throw Error.ledgerOutOfSync(ledgerRows: m, expected: total)
        }

        // 4. Build the per-chunk sqrt-sample training set, matching
        // Witchcraft's `sample_embeddings_for_kmeans`.
        var rng = SplitMix64(seed: config.seed)
        var trainingRows: [[Float]] = []
        for chunkID in sortedChunkIDs {
            let tokens = ledger[chunkID]!
            let chunkM = tokens.count
            let sampleK = Int(Double(chunkM).squareRoot().rounded(.up))
            let picked = Self.sampleDistinct(
                count: min(sampleK, chunkM), from: chunkM, rng: &rng
            )
            for i in picked {
                trainingRows.append(tokens[i])
            }
        }
        let trainM = trainingRows.count

        // 5. Compute k via Witchcraft's formula, clamped to the training
        // set size and to >= 1.
        var k = Int((config.kCoefficient * Double(m).squareRoot()).rounded())
        k = max(k, 1)
        if trainM < k {
            k = max(trainM / 4, 1)
        }
        k = min(k, trainM)

        // 6. Run k-means on the training set.
        var trainFlat = [Float]()
        trainFlat.reserveCapacity(trainM * dims)
        for row in trainingRows { trainFlat.append(contentsOf: row) }
        let kmeansResult = KMeans.cluster(
            data: trainFlat, dims: dims, clusters: k,
            maxIterations: config.kmeansIterations,
            rng: &rng
        )
        let centroids = kmeansResult.centroids

        // 7. Assign every row in [minChunkID, maxChunkID] to its nearest
        // centroid using the trained centers.
        var allFlat = [Float]()
        allFlat.reserveCapacity(m * dims)
        for row in allRows { allFlat.append(contentsOf: row) }
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
                let row = allRows[rowIdx]
                for j in 0..<dims {
                    residualValues.append(row[j] - centerSlice[j])
                }
            }

            let bucket = BucketRecord(
                generationID: insertedGen.id,
                center: Self.encodeFloat32LE(centerSlice),
                indices: IndicesCodec.encode(pairs),
                residuals: Q4Codec.encodeResiduals(residualValues)
            )
            _ = try await storage.insertBucket(bucket)
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
    }

    /// Wipe every persisted generation (and its buckets, via FK
    /// cascade) and reset the in-memory L0 buffer. Documents and chunks
    /// are not touched.
    public func clearIndex() async throws {
        let gens = try await storage.generations()
        for g in gens {
            try await storage.deleteGeneration(id: g.id)
        }
        ledger.removeAll()
        dims = nil
        pendingMinChunkID = nil
        pendingMaxChunkID = nil
        pendingCount = 0
    }

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
