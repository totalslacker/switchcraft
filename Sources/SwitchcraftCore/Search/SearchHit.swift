import Foundation

/// One scored document returned by the search pipeline.
///
/// `uuid` identifies the document; `score` is the MaxSim-mean across query
/// tokens after document-level aggregation. Higher is better. Hits are
/// returned sorted by `score` descending with ties broken on `uuid`
/// lexicographic ascending.
public struct SearchHit: Sendable, Hashable {
    public var uuid: String
    public var score: Float

    public init(uuid: String, score: Float) {
        self.uuid = uuid
        self.score = score
    }
}
