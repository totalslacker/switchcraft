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

    /// Build a `SearchConfig`. All arguments default to upstream
    /// Witchcraft's reference constants.
    public init(
        k: Int = 32,
        tPrime: Int = 40_000,
        threshold: Float = 0,
        searchDeadline: Duration = .seconds(5)
    ) {
        precondition(k > 0, "k must be positive")
        precondition(tPrime > 0, "tPrime must be positive")
        self.k = k
        self.tPrime = tPrime
        self.threshold = threshold
        self.searchDeadline = searchDeadline
    }
}
