import Foundation

/// A value that can be bound to a SQLite prepared statement parameter.
enum SQLValue: Sendable, Hashable {
    case null
    case int(Int64)
    case real(Double)
    case text(String)
    case blob(Data)
}
