// SPDX-License-Identifier: Apache-2.0
import Foundation
import SQLite3

/// SQLite asks callers to use a sentinel destructor pointer for transient
/// buffers that it should copy. The C macro is `(void*)-1`; in Swift we
/// recreate it with `unsafeBitCast`.
internal let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// Mutable state shared between `SQLiteConnection` and the C progress
/// handler callback. Heap-allocated so a stable pointer can be passed to
/// `sqlite3_progress_handler`. Mutated only from inside the owning
/// `SQLiteStorage` actor (which serialises all SQLite access), so there
/// is no concurrent access even though the C callback fires on the same
/// thread as `sqlite3_step`.
final class ProgressHandlerState {
    /// When `false` the callback returns 0 immediately (no-op). Set to
    /// `true` only while a search call is in-flight with a deadline.
    /// Disarming via this flag ensures that write operations issued after
    /// a timed-out search are not interrupted by a stale budget value.
    var isActive: Bool = false
    var sqlPhaseStart: ContinuousClock.Instant = ContinuousClock.now
    var remainingBudget: Duration = .seconds(86_400)
}

/// C-compatible progress callback invoked every N SQLite VM instructions.
/// Returns 0 to continue, 1 to request `SQLITE_INTERRUPT`.
private func sqliteProgressCallback(_ ctx: UnsafeMutableRawPointer?) -> Int32 {
    guard let ctx else { return 0 }
    let state = Unmanaged<ProgressHandlerState>.fromOpaque(ctx).takeUnretainedValue()
    guard state.isActive else { return 0 }
    return (ContinuousClock.now - state.sqlPhaseStart) >= state.remainingBudget ? 1 : 0
}

/// Thin wrapper around a sqlite3 handle. Not thread-safe by itself; only
/// touched from inside the owning `SQLiteStorage` actor.
final class SQLiteConnection {
    private var handle: OpaquePointer?

    /// Shared state between the connection and the C progress callback.
    /// Remains valid for the entire lifetime of the connection. Accessed
    /// by `SQLiteStorage` to configure per-search deadlines.
    let progressState: ProgressHandlerState

    /// Retained opaque pointer passed to `sqlite3_progress_handler`. One
    /// extra retain is held beyond `progressState`'s ARC count; released
    /// in `deinit` to balance the `passRetained` in `init`.
    private let progressStatePtr: UnsafeMutableRawPointer

    init(path: String) throws {
        // Initialise all stored properties before any potential throw so
        // that `deinit` is always called and can release resources.
        let state = ProgressHandlerState()
        self.progressState = state
        self.progressStatePtr = Unmanaged.passRetained(state).toOpaque()
        self.handle = nil

        var h: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &h, flags, nil)
        if rc != SQLITE_OK {
            let message = h.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            sqlite3_close_v2(h)
            throw SQLiteError(code: rc, message: message)
        }
        self.handle = h
        // N = 10_000 VM instructions ≈ 5–20 ms at typical SQLite throughput.
        // This granularity is precise enough for a 5-second deadline with
        // negligible overhead. See ADR 019.
        sqlite3_progress_handler(h, 10_000, sqliteProgressCallback, progressStatePtr)
    }

    deinit {
        sqlite3_close_v2(handle)
        // Release the extra retain added by `passRetained` in `init`.
        Unmanaged<ProgressHandlerState>.fromOpaque(progressStatePtr).release()
    }

    func execute(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &err)
        if rc != SQLITE_OK {
            let message = err.map { String(cString: $0) } ?? "exec failed"
            sqlite3_free(err)
            throw SQLiteError(code: rc, message: message)
        }
    }

    func prepare(_ sql: String) throws -> SQLiteStatement {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK, let stmt else {
            throw SQLiteError(code: rc, message: errorMessage())
        }
        return SQLiteStatement(handle: stmt, owner: self)
    }

    func transaction<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN")
        do {
            let result = try body()
            try execute("COMMIT")
            return result
        } catch {
            try? execute("ROLLBACK")
            throw error
        }
    }

    func errorMessage() -> String {
        handle.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
    }

    var lastInsertRowID: Int64 {
        sqlite3_last_insert_rowid(handle)
    }
}

/// Wraps a prepared statement. Resets bindings on each `bind(_:)`.
final class SQLiteStatement {
    private var handle: OpaquePointer?
    private weak var owner: SQLiteConnection?

    init(handle: OpaquePointer, owner: SQLiteConnection) {
        self.handle = handle
        self.owner = owner
    }

    deinit {
        sqlite3_finalize(handle)
    }

    func bind(_ values: [SQLValue]) throws {
        sqlite3_reset(handle)
        sqlite3_clear_bindings(handle)
        for (i, value) in values.enumerated() {
            let idx = Int32(i + 1)
            let rc: Int32
            switch value {
            case .null:
                rc = sqlite3_bind_null(handle, idx)
            case .int(let v):
                rc = sqlite3_bind_int64(handle, idx, v)
            case .real(let v):
                rc = sqlite3_bind_double(handle, idx, v)
            case .text(let s):
                rc = sqlite3_bind_text(handle, idx, s, -1, SQLITE_TRANSIENT)
            case .blob(let data):
                rc = data.withUnsafeBytes { buf -> Int32 in
                    if buf.isEmpty {
                        return sqlite3_bind_zeroblob(handle, idx, 0)
                    }
                    return sqlite3_bind_blob(
                        handle, idx,
                        buf.baseAddress, Int32(buf.count),
                        SQLITE_TRANSIENT
                    )
                }
            }
            if rc != SQLITE_OK {
                throw SQLiteError(code: rc, message: owner?.errorMessage() ?? "bind failed")
            }
        }
    }

    /// Step the statement. Returns `true` if a row is available.
    @discardableResult
    func step() throws -> Bool {
        let rc = sqlite3_step(handle)
        switch rc {
        case SQLITE_ROW:
            return true
        case SQLITE_DONE:
            return false
        default:
            throw SQLiteError(code: rc, message: owner?.errorMessage() ?? "step failed")
        }
    }

    func reset() {
        sqlite3_reset(handle)
    }

    // MARK: - Column accessors

    func columnInt64(_ index: Int32) -> Int64 { sqlite3_column_int64(handle, index) }
    func columnDouble(_ index: Int32) -> Double { sqlite3_column_double(handle, index) }

    func columnText(_ index: Int32) -> String {
        guard let p = sqlite3_column_text(handle, index) else { return "" }
        return String(cString: p)
    }

    func columnBlob(_ index: Int32) -> Data {
        let len = Int(sqlite3_column_bytes(handle, index))
        guard len > 0, let bytes = sqlite3_column_blob(handle, index) else { return Data() }
        return Data(bytes: bytes, count: len)
    }
}
