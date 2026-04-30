// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore
import Switchcraft

extension SwitchcraftStore {
    /// Open (or reopen) a `SwitchcraftStore` backed by SQLite at the
    /// given path. Pass `":memory:"` for an ephemeral store.
    ///
    /// Reopening an existing database resumes its documents and the
    /// persisted LSM index. Adding new documents after reopen risks
    /// `Indexer.Error.ledgerOutOfSync` on the next flush — see the
    /// `SwitchcraftStore.add` docs.
    public static func sqlite(
        databasePath: String,
        embedder: any Embedder,
        config: StoreConfig = .default
    ) async throws -> SwitchcraftStore {
        let storage = SQLiteStorage(path: databasePath)
        return try await SwitchcraftStore(
            storage: storage,
            embedder: embedder,
            config: config
        )
    }
}
