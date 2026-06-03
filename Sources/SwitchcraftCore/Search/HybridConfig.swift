// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Tuneable parameters for `SearchEngine.searchHybrid`.
///
/// Hybrid search runs the vector pipeline and the FTS pipeline
/// sequentially (parallelism is intentionally avoided so the fused output
/// is byte-identical for the same inputs), pulls `perSourceBudget`
/// candidates from each, and fuses the two ranked lists via standard
/// equal-weight Reciprocal Rank Fusion:
///
/// `rrf(uuid) = Σ 1 / (rrfK + rank_in_source)`
///
/// where `rank_in_source` is **1-indexed** within each source. A document
/// absent from a source contributes 0 to the sum.
///
/// Defaults:
/// - `rrfK = 60` is the canonical Cormack & Clarke (SIGIR 2009) RRF
///   constant; it de-emphasises tail ranks without pinning all weight to
///   the head.
/// - `perSourceBudget = 50` covers typical `topK ≤ 10` use cases while
///   keeping the union/filter pass bounded to ~100 document lookups.
///
/// Switchcraft uses **equal-weight RRF**. Witchcraft's reference Rust
/// implementation applies an asymmetric `+3` rank offset to the FTS
/// source; see `adrs/008-hybrid-fusion.md` for the deviation rationale.
public struct HybridConfig: Sendable, Hashable {

    /// RRF smoothing constant. Higher values flatten the rank-weight
    /// curve. Cormack & Clarke (SIGIR 2009) use 60; this is also the
    /// most common default in modern hybrid retrieval systems.
    public var rrfK: Int

    /// Number of candidates pulled from each source before fusion.
    /// Must be at least as large as the desired `topK` for the fusion
    /// to consider documents that rank below `topK` in one source but
    /// might be lifted by the other.
    public var perSourceBudget: Int

    /// BM25 per-column weight applied to the FTS `title` column relative
    /// to the `body` column (which always has weight 1.0). A value of 3.0
    /// means a title match contributes 3× the BM25 signal of an equivalent
    /// body match. Set to 1.0 to treat title and body equally; set to 0.0
    /// to disable title boosting entirely.
    ///
    /// See ADR 026 for the FTS column ordering constraint.
    public var ftsTitleWeight: Float

    /// Build a `HybridConfig`. Defaults match the canonical RRF
    /// constant (60) and a per-source budget large enough for `topK ≤ 10`.
    public init(rrfK: Int = 60, perSourceBudget: Int = 50, ftsTitleWeight: Float = 3.0) {
        precondition(rrfK > 0, "rrfK must be positive")
        precondition(perSourceBudget > 0, "perSourceBudget must be positive")
        precondition(ftsTitleWeight >= 0 && ftsTitleWeight.isFinite, "ftsTitleWeight must be non-negative and finite")
        self.rrfK = rrfK
        self.perSourceBudget = perSourceBudget
        self.ftsTitleWeight = ftsTitleWeight
    }
}
