// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One scored document returned by the hybrid (vector + FTS) search
/// pipeline.
///
/// `score` is the fused RRF score (higher is better). Per-source rank and
/// raw score are surfaced so callers can explain the fusion or apply
/// downstream re-ranking. A `nil` rank/score means the document did not
/// appear in that source's top-`perSourceBudget` candidates.
///
/// Hits are returned sorted by `(score DESC, uuid ASC)` — see
/// `adrs/008-hybrid-fusion.md`.
public struct HybridHit: Sendable, Hashable {
    /// Document UUID, as supplied to `SwitchcraftStore.add(id:body:)`.
    public var uuid: String
    /// Fused RRF score (higher is better).
    public var score: Float
    /// 1-indexed rank in the vector pipeline, or `nil` if the document
    /// did not appear in the vector source's top candidates.
    public var vectorRank: Int?
    /// Raw MaxSim score from the vector pipeline, or `nil` if absent.
    public var vectorScore: Float?
    /// 1-indexed rank in the FTS/BM25 pipeline, or `nil` if absent.
    public var ftsRank: Int?
    /// Raw BM25 score from the FTS pipeline, or `nil` if absent.
    public var ftsScore: Float?

    /// Build a `HybridHit`. Per-source rank and score arguments are
    /// optional so callers can construct mock fixtures.
    public init(
        uuid: String,
        score: Float,
        vectorRank: Int? = nil,
        vectorScore: Float? = nil,
        ftsRank: Int? = nil,
        ftsScore: Float? = nil
    ) {
        self.uuid = uuid
        self.score = score
        self.vectorRank = vectorRank
        self.vectorScore = vectorScore
        self.ftsRank = ftsRank
        self.ftsScore = ftsScore
    }
}
