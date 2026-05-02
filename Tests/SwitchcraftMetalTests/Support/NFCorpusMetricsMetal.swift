// SPDX-License-Identifier: Apache-2.0
//
// NFCorpus metrics + fixture loaders for the SwitchcraftMetal NDCG@10
// parity gate. Issue #65.
//
// **Duplicate of the private helpers in
// `Tests/SwitchcraftTests/NFCorpusBenchmarkTests.swift`** (`loadCorpus`,
// `loadQueries`, `loadQrels`, `ndcg10`, `NFCorpusFixtureError`). They
// were file-private in the CoreML benchmark file; here they're lifted to
// `internal` so the (asset-gated) Metal benchmark can reuse them across
// suites in this target. If a third consumer arrives, extract a shared
// `MetalTestSupport` target both test targets depend on.

import Foundation

struct NFCorpusRowMetal {
    let docId: String
    let text: String
}

/// Parse `collection_map.json`: `{ "numeric_id": "MED-*" }`.
/// Witchcraft's pre-staged corpus TSV uses numeric row indices; the map
/// translates them to the `MED-*` IDs used in qrels.
func loadCollectionMapMetal(from url: URL) throws -> [String: String] {
    let data = try Data(contentsOf: url)
    guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: String] else {
        throw NFCorpusFixtureErrorMetal.qrelsNotNestedDict
    }
    return dict
}

/// Parse the corpus TSV. Witchcraft's pre-staged format is two columns:
/// `numeric_id \t text`. The numeric_id is mapped to a `MED-*` doc ID via
/// the `collectionMap` from `collection_map.json` (required for qrel lookup).
func loadCorpusMetal(from url: URL, collectionMap: [String: String]) throws -> [NFCorpusRowMetal] {
    let raw = try String(contentsOf: url, encoding: .utf8)
    var rows: [NFCorpusRowMetal] = []
    for line in raw.split(separator: "\n", omittingEmptySubsequences: true) {
        let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard cols.count >= 2 else { continue }
        let numericId = String(cols[0])
        let text = String(cols[1...].joined(separator: "\t"))
        if numericId.isEmpty || text.isEmpty { continue }
        guard let docId = collectionMap[numericId] else { continue }
        rows.append(NFCorpusRowMetal(docId: docId, text: text))
    }
    return rows
}

/// Parse the queries TSV. Columns: `query-id \t query`.
func loadQueriesMetal(from url: URL) throws -> [(queryId: String, text: String)] {
    let raw = try String(contentsOf: url, encoding: .utf8)
    var queries: [(queryId: String, text: String)] = []
    for line in raw.split(separator: "\n", omittingEmptySubsequences: true) {
        let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard cols.count >= 2 else { continue }
        let qid = String(cols[0])
        let text = String(cols[1...].joined(separator: "\t"))
        if qid.isEmpty || text.isEmpty { continue }
        queries.append((queryId: qid, text: text))
    }
    return queries
}

/// Parse pytrec_eval-style qrels: `{ qid: { docid: grade } }`. Grades
/// are integers 0–3 in NFCorpus.
func loadQrelsMetal(from url: URL) throws -> [String: [String: Int]] {
    let data = try Data(contentsOf: url)
    let parsed = try JSONSerialization.jsonObject(with: data, options: [])
    guard let topDict = parsed as? [String: Any] else {
        throw NFCorpusFixtureErrorMetal.qrelsNotNestedDict
    }
    var result: [String: [String: Int]] = [:]
    result.reserveCapacity(topDict.count)
    for (qid, value) in topDict {
        guard let inner = value as? [String: Any] else {
            throw NFCorpusFixtureErrorMetal.qrelsNotNestedDict
        }
        var grades: [String: Int] = [:]
        grades.reserveCapacity(inner.count)
        for (docId, gradeAny) in inner {
            if let g = gradeAny as? NSNumber {
                grades[docId] = g.intValue
            } else if let s = gradeAny as? String, let g = Int(s) {
                grades[docId] = g
            } else {
                throw NFCorpusFixtureErrorMetal.qrelsGradeNotInt(qid: qid, docId: docId)
            }
        }
        result[qid] = grades
    }
    return result
}

/// Standard IR NDCG@10:
///   DCG = Σ_{i=1..10} (2^rel_i − 1) / log₂(i + 1)
///   IDCG = DCG of the top-10 relevant grades sorted descending
///   NDCG = DCG / IDCG, returning 0 when IDCG is 0 (no relevant docs).
func ndcg10Metal(retrieved: [String], relevant: [String: Int]) -> Double {
    var dcg = 0.0
    for (i, docId) in retrieved.prefix(10).enumerated() {
        let grade = relevant[docId] ?? 0
        if grade > 0 {
            dcg += (pow(2.0, Double(grade)) - 1.0) / log2(Double(i + 2))
        }
    }
    let idealGrades = relevant.values.sorted(by: >).prefix(10)
    var idcg = 0.0
    for (i, grade) in idealGrades.enumerated() {
        idcg += (pow(2.0, Double(grade)) - 1.0) / log2(Double(i + 2))
    }
    guard idcg > 0 else { return 0 }
    return dcg / idcg
}

enum NFCorpusFixtureErrorMetal: Error, CustomStringConvertible {
    case qrelsNotNestedDict
    case qrelsGradeNotInt(qid: String, docId: String)

    var description: String {
        switch self {
        case .qrelsNotNestedDict:
            return "qrels.test.json must be a nested-dict { qid: { docid: grade } } — see scripts/fetch-nfcorpus.sh"
        case .qrelsGradeNotInt(let qid, let docId):
            return "qrels.test.json grade for (\(qid), \(docId)) is not an integer"
        }
    }
}
