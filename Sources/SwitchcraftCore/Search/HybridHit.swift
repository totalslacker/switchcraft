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
    public var uuid: String
    public var score: Float
    public var vectorRank: Int?
    public var vectorScore: Float?
    public var ftsRank: Int?
    public var ftsScore: Float?

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
