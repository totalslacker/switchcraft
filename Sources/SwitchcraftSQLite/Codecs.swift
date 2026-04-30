// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Helpers for encoding/decoding the few non-primitive fields that travel
/// through SQLite as text or real columns.
enum Codecs {
    /// Encode `[Int]` as a comma-separated list of decimal integers.
    static func encode(_ ints: [Int]) -> String {
        ints.map(String.init).joined(separator: ",")
    }

    /// Decode `[Int]` from a comma-separated list. Empty string → empty array.
    static func decodeInts(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        return text.split(separator: ",").compactMap { Int($0) }
    }

    /// Encode `Date` as Unix epoch seconds (REAL).
    static func encode(_ date: Date) -> Double {
        date.timeIntervalSince1970
    }

    /// Decode `Date` from Unix epoch seconds.
    static func decodeDate(_ seconds: Double) -> Date {
        Date(timeIntervalSince1970: seconds)
    }

    /// Build a JSON path string for `json_extract(metadata, ?)` lookups,
    /// quoting and escaping the key so any character is safe.
    static func jsonPath(forKey key: String) -> String {
        let escaped = key
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "$.\"\(escaped)\""
    }
}
