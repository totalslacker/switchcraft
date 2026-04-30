// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Tunable parameters for `Indexer`.
///
/// The two primary knobs (`l0Capacity`, `lsmFanout`) match upstream
/// Witchcraft's `L0_CAPACITY` and `LSM_FANOUT` constants. Per-level
/// capacity is derived as `l0Capacity * lsmFanout^(level + 1)` — same
/// formula as `level_capacity` in `lib.rs`.
///
/// `kmeansIterations` and `kCoefficient` are the k-means knobs. Defaults
/// match Witchcraft's `run_kmeans_for_index` (5 iterations, k = round(16
/// * sqrt(n))). Override these only with strong reason — they affect
/// every persisted bucket.
///
/// `seed` seeds the indexer's `SplitMix64` for k-means initialisation
/// and per-chunk training-set sampling. Same seed + same input order +
/// same config ⇒ byte-identical bucket blobs across builds.
public struct IndexerConfig: Sendable, Hashable {

    /// Threshold (in embeddings, not chunks) at which an L0 cascade may
    /// fire. Witchcraft production: 1024.
    public var l0Capacity: Int

    /// Per-level fan-out exponent. Witchcraft production: 16.
    public var lsmFanout: Int

    /// Number of k-means iterations. Witchcraft uses 5 in
    /// `run_kmeans_for_index` (overrides the library default of ~25).
    public var kmeansIterations: Int

    /// Coefficient in the k formula `k = round(coefficient * sqrt(n))`.
    /// Witchcraft uses 16.0.
    public var kCoefficient: Double

    /// Seed for the indexer's deterministic `SplitMix64`.
    public var seed: UInt64

    public init(
        l0Capacity: Int,
        lsmFanout: Int,
        kmeansIterations: Int = 5,
        kCoefficient: Double = 16.0,
        seed: UInt64 = 42
    ) {
        precondition(l0Capacity > 0, "l0Capacity must be positive")
        precondition(lsmFanout > 1, "lsmFanout must be >= 2 to make cascade meaningful")
        precondition(kmeansIterations > 0, "kmeansIterations must be positive")
        precondition(kCoefficient > 0, "kCoefficient must be positive")
        self.l0Capacity = l0Capacity
        self.lsmFanout = lsmFanout
        self.kmeansIterations = kmeansIterations
        self.kCoefficient = kCoefficient
        self.seed = seed
    }

    /// Production defaults matching upstream Witchcraft.
    public static let production = IndexerConfig(
        l0Capacity: 1024,
        lsmFanout: 16
    )

    /// Reduced-threshold config for tests that need to trigger cascades
    /// without inserting thousands of embeddings. Mirrors Witchcraft's
    /// `cfg(test)` constants (`L0_CAPACITY = 4`, `LSM_FANOUT = 2`).
    public static func testing(
        l0Capacity: Int = 4,
        lsmFanout: Int = 2,
        seed: UInt64 = 42
    ) -> IndexerConfig {
        IndexerConfig(
            l0Capacity: l0Capacity,
            lsmFanout: lsmFanout,
            seed: seed
        )
    }

    /// Capacity (in embeddings) for the given level.
    /// Returns `l0Capacity * lsmFanout^(level + 1)`.
    public func levelCapacity(_ level: Int) -> Int {
        precondition(level >= 0)
        var cap = l0Capacity
        for _ in 0...level {
            cap *= lsmFanout
        }
        return cap
    }
}
