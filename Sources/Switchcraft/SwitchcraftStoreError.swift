// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Errors thrown by `SwitchcraftStore`.
public enum SwitchcraftStoreError: Error, Equatable, Sendable {
    /// A method was called after `shutdown()` returned. The store cannot
    /// be reused — open a new `SwitchcraftStore` against the same backing
    /// storage instead.
    case alreadyShutDown

    /// The supplied embedder reported a `dims` value that the engine
    /// cannot accept. `dims` must be positive and even (the Q4 residual
    /// codec packs two nibbles per byte).
    case invalidEmbeddingDimensions(Int)

    /// The embedder returned a buffer whose length is not a multiple of
    /// `dims`. Indicates a buggy `Embedder` implementation.
    case embeddingMismatch(count: Int, dims: Int)

    /// The search exceeded its configured wall-time deadline. `elapsed`
    /// is total time since `search()` was entered — it is always ≥ the
    /// configured deadline (within one progress-handler tick for the
    /// SQLite-interrupt path). `partialTiming` carries best-effort
    /// per-stage durations for whichever stages completed before the
    /// deadline fired (issue #148) — later stages are `nil`, not zero.
    case searchTimedOut(elapsed: Duration, partialTiming: SearchTiming)
}
