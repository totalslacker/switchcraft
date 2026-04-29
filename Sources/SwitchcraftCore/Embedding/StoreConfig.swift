import Foundation

/// Aggregate configuration for `SwitchcraftStore`.
///
/// Bundles the `Indexer`, `SearchEngine`, and hybrid-fusion configs so
/// callers can tune the public store without reaching into the engines.
public struct StoreConfig: Sendable, Hashable {
    public var indexer: IndexerConfig
    public var search: SearchConfig
    public var hybrid: HybridConfig

    public init(
        indexer: IndexerConfig = .production,
        search: SearchConfig = SearchConfig(),
        hybrid: HybridConfig = HybridConfig()
    ) {
        self.indexer = indexer
        self.search = search
        self.hybrid = hybrid
    }

    /// Production defaults — Witchcraft-parity indexer, k=32 / tPrime=40k
    /// search, RRF rrfK=60 / perSourceBudget=50.
    public static let `default` = StoreConfig()

    /// Reduced-threshold config for tests that need to exercise cascades
    /// without inserting thousands of embeddings. Mirrors
    /// `IndexerConfig.testing()`.
    public static func testing(
        indexer: IndexerConfig = .testing(),
        search: SearchConfig = SearchConfig(),
        hybrid: HybridConfig = HybridConfig()
    ) -> StoreConfig {
        StoreConfig(indexer: indexer, search: search, hybrid: hybrid)
    }
}
