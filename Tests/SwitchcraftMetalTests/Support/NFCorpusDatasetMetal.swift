// SPDX-License-Identifier: Apache-2.0
//
// NFCorpus dataset resolver for the SwitchcraftMetal NDCG@10 parity gate
// (`NFCorpusMetalBenchmarkTests`). Issue #65.
//
// **Duplicate of `Tests/SwitchcraftTests/Support/NFCorpusDataset.swift`**
// per the issue #65 plan §"Key Decisions" — `SwitchcraftMetalTests` does
// not depend on `SwitchcraftTests` so the resolver needs to live here
// too. If a third consumer arrives, extract a shared `MetalTestSupport`
// target both depend on.

import Foundation

enum NFCorpusDatasetMetal {
    /// Environment variable consulted by tests. Documented in README.md.
    static let envVar = "SWITCHCRAFT_NFCORPUS_DIR"

    /// Plaintext TSV: `numeric_id \t text` per row (Witchcraft pre-staged format).
    static let corpusFileName = "nfcorpus.tsv"
    /// Plaintext TSV: `query-id \t query` per row.
    static let queriesFileName = "questions.test.tsv"
    /// pytrec_eval-style nested JSON: `{ qid: { docid: grade } }`.
    static let qrelsFileName = "qrels.test.json"
    /// JSON: `{ "numeric_id": "MED-*" }` — maps corpus row indices to qrel doc IDs.
    static let collectionMapFileName = "collection_map.json"

    /// `true` iff the env var is set and all four required files exist
    /// under the resolved directory.
    static var isAvailable: Bool { url != nil }

    /// Resolved directory URL, or `nil` if the env var is unset / the
    /// directory or any required file is missing.
    static var url: URL? {
        guard
            let raw = ProcessInfo.processInfo.environment[envVar],
            !raw.isEmpty
        else {
            return nil
        }
        let dir = URL(fileURLWithPath: raw, isDirectory: true)
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        for name in [corpusFileName, queriesFileName, qrelsFileName, collectionMapFileName] {
            let f = dir.appendingPathComponent(name)
            if !fm.fileExists(atPath: f.path) {
                return nil
            }
        }
        return dir
    }

    static var corpusURL: URL? { url?.appendingPathComponent(corpusFileName) }
    static var queriesURL: URL? { url?.appendingPathComponent(queriesFileName) }
    static var qrelsURL: URL? { url?.appendingPathComponent(qrelsFileName) }
    static var collectionMapURL: URL? { url?.appendingPathComponent(collectionMapFileName) }
}
