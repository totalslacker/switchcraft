// SPDX-License-Identifier: Apache-2.0

/// The outcome of a WAL checkpoint operation.
public enum CheckpointResult: Sendable, Hashable {
    /// The WAL was fully checkpointed and truncated to zero frames. All
    /// pending writes are durable in the main database file.
    case complete

    /// Truncation was blocked by a reader that held an open WAL snapshot.
    /// All frames were copied to the main database file (data is durable),
    /// but the WAL file could not be truncated. The consumer may retry after
    /// the reader connection closes.
    ///
    /// - Parameter framesRemaining: total WAL frames still present on disk.
    case partial(framesRemaining: Int)
}
