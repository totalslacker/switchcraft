import Foundation

/// An error from the SQLite C API.
public struct SQLiteError: Error, CustomStringConvertible, Sendable {
    public let code: Int32
    public let message: String

    public var description: String {
        "SQLite error \(code): \(message)"
    }
}
