// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One scored document returned by the search pipeline.
///
/// `uuid` identifies the document; `score` is the MaxSim-mean across query
/// tokens after document-level aggregation. Higher is better. Hits are
/// returned sorted by `score` descending with ties broken on `uuid`
/// lexicographic ascending.
public struct SearchHit: Sendable, Hashable {
    /// Document UUID, as supplied to `SwitchcraftStore.add(id:body:)`.
    public var uuid: String
    /// MaxSim-mean score across query tokens. Higher is better.
    public var score: Float

    /// Build a `SearchHit` from a UUID and score.
    public init(uuid: String, score: Float) {
        self.uuid = uuid
        self.score = score
    }
}
