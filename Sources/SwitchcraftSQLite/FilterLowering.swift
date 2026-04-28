import Foundation
import SwitchcraftCore

/// A lowered SQL fragment plus its bind parameters, suitable for splicing
/// into a `WHERE` clause.
struct WhereClause {
    var sql: String
    var bindings: [SQLValue]

    static let alwaysTrue = WhereClause(sql: "1=1", bindings: [])
    static let alwaysFalse = WhereClause(sql: "0=1", bindings: [])
}

extension StorageFilter {
    /// Lower this filter to a SQL fragment that references columns of the
    /// `document` table aliased as `d` (for queries that join `document`),
    /// or unaliased (for direct queries on `document`). Pass `tableAlias`
    /// accordingly.
    func lower(tableAlias: String? = nil) -> WhereClause {
        let prefix = tableAlias.map { "\($0)." } ?? ""

        switch self {
        case .all:
            return .alwaysTrue

        case .uuidIn(let uuids):
            if uuids.isEmpty { return .alwaysFalse }
            let placeholders = Array(repeating: "?", count: uuids.count).joined(separator: ", ")
            let bindings = uuids.map { SQLValue.text($0) }
            return WhereClause(
                sql: "\(prefix)uuid IN (\(placeholders))",
                bindings: bindings
            )

        case .dateRange(let start, let end):
            var parts: [String] = []
            var bindings: [SQLValue] = []
            if let start {
                parts.append("\(prefix)date >= ?")
                bindings.append(.real(Codecs.encode(start)))
            }
            if let end {
                parts.append("\(prefix)date <= ?")
                bindings.append(.real(Codecs.encode(end)))
            }
            if parts.isEmpty { return .alwaysTrue }
            return WhereClause(sql: parts.joined(separator: " AND "), bindings: bindings)

        case .metadataEquals(let key, let value):
            let path = Codecs.jsonPath(forKey: key)
            return WhereClause(
                sql: "json_extract(CAST(\(prefix)metadata AS TEXT), ?) = ?",
                bindings: [.text(path), .text(value)]
            )

        case .and(let filters):
            if filters.isEmpty { return .alwaysTrue }
            var bindings: [SQLValue] = []
            let parts = filters.map { (sub: StorageFilter) -> String in
                let lowered = sub.lower(tableAlias: tableAlias)
                bindings.append(contentsOf: lowered.bindings)
                return "(\(lowered.sql))"
            }
            return WhereClause(sql: parts.joined(separator: " AND "), bindings: bindings)

        case .or(let filters):
            if filters.isEmpty { return .alwaysFalse }
            var bindings: [SQLValue] = []
            let parts = filters.map { (sub: StorageFilter) -> String in
                let lowered = sub.lower(tableAlias: tableAlias)
                bindings.append(contentsOf: lowered.bindings)
                return "(\(lowered.sql))"
            }
            return WhereClause(sql: parts.joined(separator: " OR "), bindings: bindings)

        case .not(let filter):
            let inner = filter.lower(tableAlias: tableAlias)
            return WhereClause(sql: "NOT (\(inner.sql))", bindings: inner.bindings)
        }
    }
}
