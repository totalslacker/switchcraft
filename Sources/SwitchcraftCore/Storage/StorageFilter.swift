import Foundation

/// Backend-agnostic filter expression applied to document queries.
///
/// Backends are responsible for lowering this expression to their native
/// query form (SQL `WHERE`, KV scans, etc.). The expression is intentionally
/// small — extend with new cases as concrete needs appear, not speculatively.
public indirect enum StorageFilter: Sendable, Hashable {
    /// Match every document.
    case all

    /// Match documents whose uuid is in the given set.
    case uuidIn(Set<String>)

    /// Match documents whose date falls within the (inclusive) range.
    /// Either bound may be `nil` to denote open-ended.
    case dateRange(start: Date?, end: Date?)

    /// Match documents whose JSON metadata, decoded as `[String: String]`,
    /// has the given key set to the given value.
    case metadataEquals(key: String, value: String)

    /// All sub-filters must match.
    case and([StorageFilter])

    /// Any sub-filter must match.
    case or([StorageFilter])

    /// Negation.
    case not(StorageFilter)
}

extension StorageFilter {
    /// Evaluate this filter against an in-memory document. Backends that
    /// can lower the filter to a native query should do so; this helper
    /// is for the in-memory reference implementation and tests.
    public func matches(_ document: DocumentRecord) -> Bool {
        switch self {
        case .all:
            return true
        case .uuidIn(let uuids):
            return uuids.contains(document.uuid)
        case .dateRange(let start, let end):
            if let start, document.date < start { return false }
            if let end, document.date > end { return false }
            return true
        case .metadataEquals(let key, let value):
            guard
                let object = try? JSONSerialization.jsonObject(with: document.metadata),
                let dict = object as? [String: Any]
            else {
                return false
            }
            return (dict[key] as? String) == value
        case .and(let filters):
            return filters.allSatisfy { $0.matches(document) }
        case .or(let filters):
            return filters.contains { $0.matches(document) }
        case .not(let filter):
            return !filter.matches(document)
        }
    }
}
