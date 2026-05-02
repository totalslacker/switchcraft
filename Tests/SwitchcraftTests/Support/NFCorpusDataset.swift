import Foundation

/// Resolves the NFCorpus test split for the asset-gated NDCG@10 parity
/// suite (`NFCorpusBenchmarkTests`).
///
/// The NFCorpus dataset (~5–6 MB plaintext) is **not committed** to the
/// repository — its license (academic use only) is incompatible with
/// Switchcraft's Apache 2.0 release intent. To exercise the benchmark:
///
///   1. Run `scripts/fetch-nfcorpus.sh` to populate a local directory
///      with the three required plaintext files (or obtain them by any
///      other means under whatever terms you accept).
///   2. Set `SWITCHCRAFT_NFCORPUS_DIR=<absolute-path-to-directory>`.
///   3. Set `SWITCHCRAFT_XTR_MLPACKAGE` (the suite is double-gated on
///      both env vars — see `CoreMLAsset`).
///   4. `swift test --filter NFCorpusBenchmark`.
///
/// When either env var is unset or the expected files are missing, the
/// suite skips cleanly via Swift Testing's `.enabled(if:)` trait. CI
/// sets neither, so the benchmark never runs there.
enum NFCorpusDataset {
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
    static var isAvailable: Bool {
        url != nil
    }

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
