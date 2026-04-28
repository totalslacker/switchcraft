import Foundation

/// Metaspace decoder: reconstructs text from token strings by reversing the ▁ encoding.
struct MetaspaceDecoder: Sendable {
    let replacement: String      // U+2581 (▁) as String
    let stripLeading: Bool       // true when prependScheme != "never"

    init(from json: [String: Any]) {
        replacement = (json["replacement"] as? String) ?? "▁"
        // prepend_scheme takes precedence over add_prefix_space
        if let scheme = json["prepend_scheme"] as? String {
            stripLeading = scheme != "never"
        } else if let addPrefix = json["add_prefix_space"] as? Bool {
            stripLeading = addPrefix
        } else {
            stripLeading = true
        }
    }

    /// Reconstruct text from token strings: join them, replace ▁ with space, strip leading space.
    func decode(_ tokens: [String]) -> String {
        let joined = tokens.joined()
        var result = joined.replacingOccurrences(of: replacement, with: " ")
        if stripLeading, result.hasPrefix(" ") {
            result = String(result.dropFirst())
        }
        return result
    }
}
