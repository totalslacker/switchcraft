# ADR 003 — Project-wide typing conventions for parity-sensitive code

**Status**: Accepted (living document — extend in-place; cite this ADR in future PRs)  
**Date**: 2026-04-28  
**Issue**: #6 (Phase 1: Tokenizer)

This ADR is project-wide. Every Rust-to-Swift port in Switchcraft (tokenizer, codec, K-means, T5 inference, scoring kernels) is subject to these conventions. When you encounter a new parity-relevant type decision during porting, **add it to this ADR in the same PR** rather than leaving it as an implicit convention buried in code review.

Future PRs that touch parity-relevant arithmetic must cite this ADR in the commit body or PR description.

---

## (a) Floating-point precision for cross-implementation parity

**Rule**: Any accumulator or comparand whose value can flip a bit-exact-parity outcome must use `Double` (Float64), not `Float`.

**Rationale**: The upstream Witchcraft/HuggingFace Rust implementation uses `f64` for all log-probability accumulators and score comparands. `Float32` accumulates rounding error that can flip path choices at borderline comparisons, producing different token IDs or scores than the Rust reference.

**Worked example** — Viterbi `bestPathScore` in `UnigramModel.swift`: vocabulary scores are stored as `f64` in the tokenizer JSON and in the Rust model. The DP table accumulates sums of these scores. Using `Float32` here produces wrong token IDs on at least one fixture input; `Double` matches the Rust output exactly.

**Carve-outs** (stay `Float`/Float32 per upstream):
- Residual codec arithmetic (`Q4Codec`) — upstream uses `f32`; the codec's precision requirements are independent of bit-exact Rust parity.
- Embedding vectors and centroid arithmetic — stored and computed as `Float` to match CoreML tensor type.
- Anything that feeds a Metal buffer or CoreML model input must stay `Float`.

**Anything else** that is a log-probability sum, score accumulator, or score comparand: use `Double`.

---

## (b) String indexing in hot paths

**Rule**: Swift's `String.Index` and `Character` operations are O(n) per offset advance due to variable-width UTF-8/UTF-16 encoding. Any port of a Rust algorithm that indexes a string by integer position **must** materialise the input as `[UInt8]` (UTF-8) upfront and use `Int` indices throughout.

**Rationale**: Rust code indexes strings by byte position using `&[u8]` and integer offsets. A naive Swift port that calls `string.index(string.startIndex, offsetBy: i)` in a loop degrades an O(n) algorithm to O(n²).

**Worked example** — Viterbi DP in `UnigramModel.swift`: the forward pass iterates over `utf8.count` byte positions. Each iteration advances by the byte length of one Unicode scalar (1–4 bytes). Materialising `Array(token.utf8)` once and using `Int` indices costs one allocation per token; Swift character subscripting would cost O(n) per position advance.

**Rule applies to**: any future text-processing hot path that iterates by code unit position. If a loop body calls `string.index(...)` more than once, materialise the bytes instead.

---

## (c) Integer widths for IDs and offsets

**Rule**: Use the right integer width at the right boundary; do not silently widen or narrow.

| Value | Type | Rationale |
|---|---|---|
| Token IDs (public API) | `Int32` | Matches upstream Rust `u32` cast to signed, and CoreML `MLMultiArray` Int32 type |
| Token IDs (internal DP) | `Int32` | Keep consistent with public boundary; avoid narrowing conversions |
| Trie node indices, byte offsets | `Int` | SPM-idiomatic; no overflow risk for reasonable inputs |
| Backend-assigned record IDs | `Int64` | Matches existing `SwitchcraftStorage` protocol and SQLite `rowid` |
| Vocabulary size / indices | `Int` | Array subscript type; no overflow risk |

**Worked example** — `BestPathNode.tokenID` in `UnigramModel.swift` is `Int32` even though the DP table uses `Int` for positions. The conversion happens once at backtrack time, not inside the inner loop.

---

## (d) Sendable and value-semantics convention

**Rule**: Public types default to value types (`struct` / `enum`); `Sendable` conformance is required for everything that crosses an actor boundary. Reference types (`class`) only when there is a concrete reason: resource handles, large reusable mutable buffers, or reference-counted sharing.

**Rationale**: Switchcraft's public API surface is an actor (`SwitchcraftStore`). Any type passed into or returned from the actor must be `Sendable`. Value types are `Sendable` by default when all stored properties are `Sendable`.

**Worked example** — `struct Tokenizer: Sendable` (the public tokenizer type). Its stored properties are all value types (`[UInt32]`, `[UInt8]`, `Double`, `Int32`, `[String: Int32]`), so `Sendable` is satisfied without annotation. The internal `ByteTrie` struct also uses value semantics for the same reason.

**When a class is acceptable**: `DoubleArrayTrie` could be a class to avoid copy-on-write overhead for the 237 KB node array, but the copy is never triggered in practice (the trie is built once and never mutated). Keeping it a struct keeps the conformance trivial.
