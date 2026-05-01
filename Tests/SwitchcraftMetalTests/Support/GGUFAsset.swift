// SPDX-License-Identifier: Apache-2.0
//
// Asset resolution for the SwitchcraftMetal round-trip parity tests.
// Mirrors `Tests/SwitchcraftTests/Support/CoreMLAsset.swift` per
// ADR 010(d) — env-var gating with clean skip when the asset is
// missing.

import Foundation

/// Resolves the canonical Q4-quantised XTR-base-en GGUF for asset-gated
/// tests. The Q4_K asset is **not committed** to the repository
/// (~80 MB, see ADR 010(j) and ADR 016). Callers point
/// `SWITCHCRAFT_XTR_GGUF` at a local copy.
///
/// Asset acquisition pipeline is documented in
/// `docs/porting/ggml-t5.md` §"Asset acquisition" and in README.md
/// §"Metal embedder setup".
enum GGUFAsset {
    /// Environment variable consulted by tests. Documented in README.md.
    static let envVar = "SWITCHCRAFT_XTR_GGUF"

    /// `true` iff the env var is set and resolves to an existing path.
    static var isAvailable: Bool { url != nil }

    /// Resolved URL of the GGUF file, or `nil` when the env var is
    /// unset / the path does not exist.
    static var url: URL? {
        guard
            let raw = ProcessInfo.processInfo.environment[envVar],
            !raw.isEmpty
        else {
            return nil
        }
        let url = URL(fileURLWithPath: raw)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return url
    }
}

/// Resolves the optional FP32 reference dump used by the round-trip
/// parity test. Produced by `llama.cpp`'s `gguf-dump --print-data` (or
/// equivalent) on the FP32 GGUF that Witchcraft's quantizer also
/// publishes; format is a flat little-endian FP32 stream prefixed with
/// a small JSON sidecar describing tensor offsets.
///
/// The sidecar layout matches `Tests/Fixtures/xtr-base-en.embeddings.json`
/// in spirit but indexes ggml weight tensors rather than embedding rows:
/// ```
/// { "model": "google/xtr-base-en@<sha>",
///   "tensors": [
///     { "name": "encoder.embed_tokens.weight",
///       "shape": [<rows>, <cols>],
///       "byteOffset": <o>, "byteLength": <rows * cols * 4> },
///     ...
///   ] }
/// ```
///
/// When the env var is unset, asset-gated tests skip cleanly.
enum GGUFFP32Reference {
    /// Path to the JSON index file. The companion `.bin` is expected to
    /// sit alongside it with the same basename + `.bin` extension.
    static let envVar = "SWITCHCRAFT_XTR_GGUF_FP32_REF"

    static var isAvailable: Bool { indexURL != nil }

    static var indexURL: URL? {
        guard
            let raw = ProcessInfo.processInfo.environment[envVar],
            !raw.isEmpty
        else {
            return nil
        }
        let url = URL(fileURLWithPath: raw)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return url
    }

    /// Companion binary blob, derived from the index URL by swapping
    /// the extension to `.bin`.
    static var blobURL: URL? {
        guard let idx = indexURL else { return nil }
        let blob = idx.deletingPathExtension().appendingPathExtension("bin")
        guard FileManager.default.fileExists(atPath: blob.path) else {
            return nil
        }
        return blob
    }
}
