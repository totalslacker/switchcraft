// SPDX-License-Identifier: Apache-2.0
import SwitchcraftCore

/// SQLite reference backend for Switchcraft's storage protocol.
///
/// This is the MVP storage implementation. Other backends (DuckDB, RocksDB,
/// LMDB, Postgres+pgvector, in-memory) can be added behind the same protocol
/// without touching the engine layer.
///
/// See `docs/Plan.md` for the schema and protocol design.
public enum SwitchcraftSQLite {
    /// Semantic version of the package. Update with releases.
    public static let version = SwitchcraftCore.version
}
