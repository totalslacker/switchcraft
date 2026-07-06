// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Tuneable parameters for `SearchEngine.search`.
///
/// Defaults match upstream Witchcraft's `match_centroids` constants
/// (`src/lib.rs`): `k = 32` candidate centroids per query token,
/// `tPrime = 40_000` candidate-token soft budget per query token,
/// `threshold = 0` (no cut-off). See `adrs/006-search-constants.md`
/// for provenance.
public struct SearchConfig: Sendable, Hashable {

    /// Maximum number of centroids selected per query token, per LSM
    /// generation. Witchcraft uses 32.
    public var k: Int

    /// Per-query-token soft budget on the cumulative number of candidate
    /// token embeddings selected from buckets. The selection loop appends
    /// a centroid then checks the budget, so the centroid that crosses
    /// the threshold IS included. `cumsum` resets for each query token.
    /// Witchcraft uses 40 000.
    public var tPrime: Int

    /// Drop documents whose final aggregated score falls below this
    /// threshold. Witchcraft's reference value is 0 (no cut-off); raise
    /// it to filter low-confidence hits.
    public var threshold: Float

    /// Maximum wall-time budget for a single `SwitchcraftStore.search()`
    /// call, measured from the instant `search()` is entered. When
    /// elapsed time reaches this value the search is aborted with
    /// `SwitchcraftStoreError.searchTimedOut`. Default is 5 seconds.
    /// Override per-call via `SwitchcraftStore.search(deadline:)`.
    public var searchDeadline: Duration

    /// Minimum decoded surface-form length for query tokens. Token embeddings
    /// whose surface form satisfies `count <= queryMinSurfaceFormLength` are
    /// dropped from the query representation before MaxSim scoring.
    ///
    /// Default `0` means no filter — all token positions are kept, preserving
    /// Witchcraft-identical behaviour. Set to `2` to suppress single- and
    /// two-character subword fragments (e.g. `"t"`, `"le"`, `"by"`) that
    /// inflate noise scores for short proper-noun queries at corpus scale.
    ///
    /// See ADR 028 for rationale and interaction with ADR 006 MEAN aggregation.
    public var queryMinSurfaceFormLength: Int

    /// Benchmark-only escape hatch that forces `SearchEngine.search` onto
    /// the pre-#140 sequential, non-deduplicated code path
    /// (`searchLegacy`) instead of the parallel-decode + query-token-dedup
    /// path (`searchOptimized`) that is the default as of ADR 035.
    ///
    /// Not intended for production use — it exists so the search-latency
    /// benchmark can measure both code paths against the same fixture in
    /// a single test invocation. Both paths produce identical output
    /// (see ADR 006(f) and ADR 035); only wall-clock time differs.
    public var legacySequentialSearch: Bool

    /// Build a `SearchConfig`. All arguments default to upstream
    /// Witchcraft's reference constants.
    public init(
        k: Int = 32,
        tPrime: Int = 40_000,
        threshold: Float = 0,
        searchDeadline: Duration = .seconds(5),
        queryMinSurfaceFormLength: Int = 0,
        legacySequentialSearch: Bool = false
    ) {
        precondition(k > 0, "k must be positive")
        precondition(tPrime > 0, "tPrime must be positive")
        precondition(queryMinSurfaceFormLength >= 0, "queryMinSurfaceFormLength must be non-negative")
        self.k = k
        self.tPrime = tPrime
        self.threshold = threshold
        self.searchDeadline = searchDeadline
        self.queryMinSurfaceFormLength = queryMinSurfaceFormLength
        self.legacySequentialSearch = legacySequentialSearch
    }
}
