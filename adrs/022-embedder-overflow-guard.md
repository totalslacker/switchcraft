# ADR 022: Embedder overflow guard — `maxInputTokens` and `EmbedderOverflowPolicy`

**Status:** Accepted
**Date:** 2026-05-06
**Issue:** [#89](https://github.com/totalslacker/switchcraft/issues/89)

## Context

`T5CoreMLEmbedder` uses a sliding-window strategy (ADR 011) to handle inputs longer than
the CoreML model's fixed 512-token input. For a 593,285-character page that tokenises to
~148,000 tokens, `SlidingWindow.plan` generates approximately 577 windows. 577 consecutive
CoreML predictions within a single `encode()` call exhausted the ANE IOSurface buffer
pool, leaving the embedder in a state where every subsequent inference — including
small inputs — failed with `MLE5OutputPortBinder bindAndReturnError`. Process restart
was required to recover.

ADR 021 adds three reactive/proactive defence layers (autoreleasepool drainage, proactive
model reload, CPU fallback). This ADR adds the structural prevention layer: refuse to
generate a window burst this large in the first place.

## Decision

### 1. `EmbedderOverflowPolicy` and `EmbedderError` live in `SwitchcraftCore`

Both types are pure Swift with no CoreML or Metal dependency. Placing them in
`SwitchcraftCore` (not `SwitchcraftCoreML`) makes them:

- Available to any `Embedder` conformer, including `T5MetalEmbedder` and future backends.
- Reachable by callers who hold an `any Embedder` reference without downcasting.
- Not gated behind `#if canImport(CoreML)`.

### 2. `maxInputTokens: Int { get }` is added to the `Embedder` protocol

Adding the limit to the protocol rather than only to concrete types:

- Makes the safety contract queryable on any `Embedder`-typed reference.
- Forces new conformers to declare an explicit limit — an unknown embedder with no
  stated limit is unsafe for bulk-index consumers.
- Follows the existing pattern where `dims` and `modelIdentifier` are protocol-level.

Conformers are encouraged (but not required by the compiler) to implement this as a
`nonisolated let` stored property so callers can read it without entering an actor.

### 3. Default `maxInputTokens = 8 * windowSize`

For the standard 512-token window, this is 4,096 total tokens → ~15 windows at stride 256.
The multiplier form scales naturally when `windowSize` is customised at init. 4,096 covers
most real-world documents (~3,000 words) while keeping the window count well below the
threshold observed to exhaust the ANE pool (~577).

Callers running in environments with larger IOSurface budgets (or no ANE) may pass a larger
value at init. A `precondition(maxInputTokens >= windowSize)` guards against setting the
limit below one window, which would make every non-trivial encode either truncate to
padding or always throw.

### 4. Two policies: `.truncate` (default) and `.reject`

`.truncate` is the default for three reasons:
- It matches the established HuggingFace convention (`truncation=True, max_length=...`).
- The primary use case (retrieval, classification, bulk indexing) benefits from prefix
  embeddings when the full document is too long.
- It preserves backward compatibility — existing callers that do not pass `overflowPolicy`
  get silent truncation rather than a new throw path.

`.reject` is provided for callers that cannot accept silent data loss (e.g., applications
that want to split long documents themselves and know exactly which text was embedded).

### 5. `T5MetalEmbedder` gains the same guard

`T5MetalEmbedder` uses `SlidingWindow` identically to `T5CoreMLEmbedder`. While it lacks
the IOSurface failure mode, a 577-Metal-command-buffer burst from one `encode` call is both
slow and memory-intensive. With `maxInputTokens` now on the protocol, adding the guard to
the Metal embedder is both consistent and low-risk. No stub-based tests are added for the
Metal path because `T5MetalEmbedder.init` requires a real Metal device and GGUF asset;
the CoreML stub tests provide adequate algorithmic coverage of the overflow guard logic.

## Guard placement

The overflow guard is inserted in `encode(_:)` after `tokenizer.encode(text, addSpecialTokens: true)`
and before `SlidingWindow.plan`. This is the earliest point where the authoritative token
count is known. It fires inside the re-entrancy guard's `inFlight` window, so the `defer`
block that releases the next waiter still executes correctly when `.reject` throws.

## Consequences

- **Breaking change:** `Embedder` protocol gains a required property `maxInputTokens: Int { get }`.
  All existing conformers must implement it. Test mocks and local test stubs return `Int.max`.
- `T5CoreMLEmbedder` and `T5MetalEmbedder` gain two new init parameters
  (`maxInputTokens: Int? = nil`, `overflowPolicy: EmbedderOverflowPolicy = .truncate`)
  with defaults that maintain backward compatibility.
- `EmbedderError` is a new public enum in `SwitchcraftCore`. It will accumulate future
  embedder-level errors (not model-specific errors, which remain on `T5CoreMLEmbedderError`).
- Bulk-index consumers indexing very large documents will silently receive prefix embeddings
  unless they explicitly configure `.reject`. This is documented on the `encode` method.
