# ADR 009 — Public API shape (`SwitchcraftStore`, `Embedder`)

**Status**: Accepted
**Date**: 2026-04-28
**Issue**: #16 (Phase 1: SwitchcraftStore public actor)

This ADR locks the first user-facing surface of the package. Library
consumers `import Switchcraft` and reach the index through one actor
(`SwitchcraftStore`), one protocol (`Embedder`), one config struct
(`StoreConfig`), and one error enum (`SwitchcraftStoreError`). Cite this
ADR before changing any of those types' signatures.

---

## (a) One actor, one protocol, one config

`SwitchcraftStore` is the public surface. It wraps `Indexer`,
`SearchEngine`, and a `SwitchcraftStorage` backend; consumers never
touch those engines directly. Public methods:

```swift
public actor SwitchcraftStore {
    public init(storage: any SwitchcraftStorage,
                embedder: any Embedder,
                config: StoreConfig = .default) async throws
    public func add(id:date:metadata:title:body:) async throws  // title defaults to nil; see ADR 025
    public func remove(id:) async throws
    public func index() async throws
    public func clear() async throws
    public func search(query:topK:filter:) async throws -> [HybridHit]
    public func score(query:passages:) async throws -> [Float]
    public func shutdown() async throws
}
```

The store is an actor so concurrent callers serialise without ad-hoc
locks. The internal engines are also actors; the store does not add a
second layer of locking.

## (b) `Embedder` protocol contract

```swift
public protocol Embedder: Sendable {
    var dims: Int { get }
    var modelIdentifier: String { get }
    func encode(_ text: String) async throws -> [Float]
}
```

- `Sendable` because the store (an actor) holds an `Embedder` reference
  across `await` boundaries.
- `async throws` accommodates CoreML's Swift 6 surface and lets HTTP /
  RPC backends propagate failures.
- `encode` returns a flat row-major `n × dims` `[Float]`. Callers
  compute `n = result.count / dims`. We chose this over a tuple-return
  `(tokens: Int, embeddings: [Float])` because the divide-by-dims
  inference is unambiguous and keeps the protocol minimal.
- `modelIdentifier` is recorded on every persisted `ChunkRecord.model`
  so a future read can tell if the embedder used at write time differs
  from the one in use now. Pick a stable identifier (model file hash,
  semantic tag).
- Empty / whitespace-only text MAY return an empty array.
- `dims` must be positive and even — the Q4 residual codec packs two
  nibbles per byte. The store validates this once at init and throws
  `SwitchcraftStoreError.invalidEmbeddingDimensions(dims)` rather than
  letting an internal precondition crash.

## (c) `metadata: [String: String]`, JSON-encoded with `.sortedKeys`

`SwitchcraftStore.add` accepts `metadata: [String: String]`. The store
JSON-encodes the dict before storing it as `DocumentRecord.metadata: Data`
(consistent with `StorageFilter.metadataEquals(key:value:)`'s decode
path, which expects a flat `[String: String]` JSON object).

We use `JSONSerialization.WritingOptions.sortedKeys` so byte-identical
input always produces byte-identical `Data`. Without sorted keys, two
equivalent metadata dicts could produce different bytes and break the
"byte-identical determinism across runs" acceptance criterion.

## (d) Single-chunk-per-body for v1

`add(id:date:metadata:body:)` produces exactly one `ChunkRecord` per
call. `DocumentRecord.lens` is `[tokenCount]`. A `// TODO: paragraph
splitter` comment marks the chunking step; multi-chunk splitting lands
with the T5/CoreML embedder cycle.

Implication for replace / remove: when a document's body changes (or
the document is removed), the old chunk persists in storage as an
orphan. The search engine drops it because `documents(forChunkHash:)`
returns no documents whose `hash` field matches the orphan. Storage
bloat is acknowledged — the multi-chunk cycle will introduce
chunk-reference-counting.

## (e) Pre-check chunk existence to avoid double-buffering

`storage.upsertChunk` returns the existing record on hash collision but
does not signal whether an insert happened. The store therefore calls
`storage.chunk(hash:)` first; if absent, it inserts via `upsertChunk`
**and** calls `indexer.add` to buffer the embeddings. If present, it
reuses the existing chunk's id and skips `indexer.add` — calling
`indexer.add` again would double-buffer the same embeddings under the
same chunkID and break the indexer's row-count invariants on the next
flush.

**Hash computation (amended by ADR 025)**: The chunk key is
`SHA-256(embeddingText)` where `embeddingText` is `"\(title)\n\(body)"`
when `title` is non-nil, and `body` otherwise. This means two documents
with identical bodies but different titles produce distinct chunks with
their own correctly-keyed embeddings. When `title == nil`, the hash is
byte-identical to the pre-ADR-025 formula. See ADR 025 for the full
policy, store-version compatibility notes, and re-indexing guidance.

## (f) Auto-flush on `search` and `score`; no count threshold for v1

`search` and `score` call `indexer.flush()` internally, so callers who
add then immediately query never have to remember to flush. A
count-based auto-flush threshold (e.g. flush every N adds) is deferred
to v2; the explicit `index()` method remains for callers who want to
flush eagerly.

## (g) `shutdown` is idempotent and gates subsequent calls

`shutdown()` flushes pending writes, closes the storage, and sets an
internal `isShutDown` flag. Calling `shutdown` a second time is a
no-op. Every other public method calls `ensureRunning()` first and
throws `SwitchcraftStoreError.alreadyShutDown` if the flag is set.

Because Swift actors are re-entrant at `await` suspension points,
calls that started before `shutdown()` are not guaranteed to complete
normally. A public method may suspend (for example during
`indexer.flush()` or `embedder.encode(...)`), `shutdown()` may then run
and close storage, and the earlier call may resume and fail/throw.
The implementation sets the `isShutDown` flag before awaiting flush /
close so calls that *arrive* during the shutdown window observe the
flag at the next `ensureRunning()` check and throw `alreadyShutDown`,
rather than racing through gates and operating on closed storage.
There is no special "cancel mid-call" mechanism, but callers racing
`shutdown()` must tolerate either completing before shutdown takes
effect or observing shutdown-related failure after resumption.

## (h) Reopen-then-add limitation, documented only

A reopened store can serve `search` against the persisted index because
the search pipeline reads from storage. `add` after reopen risks
`Indexer.Error.ledgerOutOfSync` once the cascade walk wants to
re-cluster generations whose source embeddings were never `add`-ed in
this session. This is the upstream MVP property described on `Indexer`;
we propagate it as-is and document on `SwitchcraftStore.add`. A future
`ChunkEmbeddingCodec` (planned with the search pipeline) will lift this
limitation.

## (i) `sqlite(...)` factory ships in `SwitchcraftSQLite`, not the umbrella

`SwitchcraftStore.sqlite(databasePath:embedder:config:)` is declared as
an extension in `Sources/SwitchcraftSQLite/SwitchcraftStore+SQLite.swift`
rather than inside the `Switchcraft` umbrella. This keeps the umbrella
free of SQLite linkage — consumers of a different backend don't pay for
SQLite in their binary. Trade-off: SQLite users `import Switchcraft`
**and** `import SwitchcraftSQLite` to call `.sqlite(...)`. The factory
form follows the "Storage is pluggable" tenet.

`Package.swift` was updated to add `Switchcraft` as a target dependency
of `SwitchcraftSQLite` (transitively through `SwitchcraftCore`, which
the store needs anyway). No runtime circularity; `Switchcraft` does
not depend on `SwitchcraftSQLite`.

## (j) `MockEmbedder` deterministic recipe (test-target only)

`Tests/SwitchcraftTests/Support/MockEmbedder.swift` ships in the test
target only — not as a public type, not in any library product. The
recipe is pinned here so an implementer (or a porter writing a parity
test) can reproduce identical fixtures:

1. Trim leading/trailing whitespace from the input. Empty after trim ⇒
   return `[]`.
2. Split the trimmed text on whitespace. Empty after split ⇒ return `[]`.
3. For token index `i ∈ [0, tokenCount)`:
   - `seedString = "\(originalText)|\(i)"` (the **untrimmed** input is
     used so two strings that differ only in surrounding whitespace
     produce different vectors).
   - `digest = SHA256(seedString.utf8)`.
   - `seed = UInt64(digest[0..<8] little-endian)`.
   - Initialise a `SplitMix64(seed:)`.
   - For `j ∈ [0, dims)`: draw a `UInt64`, take its low 32 bits, then
     `scaled = Float(bits >> 8) / Float(1 << 24)` (upper 24 bits, mantissa
     -exact); `vec[j] = scaled * 2 - 1` (range `[-1, 1)`).
   - Normalise `vec` to unit L2 length. On the impossible all-zero
     draw, fall back to `[1, 0, 0, …]`.
4. Concatenate per-token vectors into the flat `[Float]` return value.

Default `dims = 128` (Witchcraft's projection dimension). The
`modelIdentifier` defaults to `"mock-embedder-v1-d\(dims)"`.

---

## Consequences

- The public surface is small and frozen. Future API changes (e.g. a
  `streamSearch`, a `topKWithSnippets`) extend rather than alter these
  signatures.
- The umbrella has zero backend dependencies beyond `SwitchcraftCore`.
  New backends drop in next to `SwitchcraftSQLite`.
- The single-chunk-per-body decision is the largest debt; it's
  load-bearing for v1 simplicity and is targeted by the splitter +
  reference-counting work that ships with the embedder cycle.
