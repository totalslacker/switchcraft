import Foundation

/// TemplateProcessing post-processor: appends special tokens per a single/pair template.
struct TemplatePostProcessor: Sendable {
    private let singleTemplate: [TemplateItem]
    private let specialTokenIDs: [String: Int32]

    enum TemplateItem: Sendable {
        case sequence
        case specialToken(String)
    }

    init(from json: [String: Any], addedTokenIDs: [String: Int32]) throws {
        // Parse single template
        guard let singleArray = json["single"] as? [[String: Any]] else {
            throw TokenizerError.missingField("post_processor.single")
        }
        var items: [TemplateItem] = []
        for item in singleArray {
            if item["Sequence"] != nil {
                items.append(.sequence)
            } else if let st = item["SpecialToken"] as? [String: Any],
                      let id = st["id"] as? String {
                items.append(.specialToken(id))
            }
        }
        singleTemplate = items

        // Build special-token ID lookup from post_processor + added_tokens
        var lookup: [String: Int32] = addedTokenIDs
        if let specialTokens = json["special_tokens"] as? [String: [String: Any]] {
            for (key, stObj) in specialTokens {
                if let ids = stObj["ids"] as? [Int], let first = ids.first {
                    lookup[key] = Int32(first)
                }
            }
        }
        specialTokenIDs = lookup
    }

    func process(_ ids: [Int32], addSpecialTokens: Bool) -> [Int32] {
        guard addSpecialTokens else { return ids }
        var result: [Int32] = []
        for item in singleTemplate {
            switch item {
            case .sequence:
                result.append(contentsOf: ids)
            case .specialToken(let name):
                if let id = specialTokenIDs[name] {
                    result.append(id)
                }
            }
        }
        return result
    }
}
