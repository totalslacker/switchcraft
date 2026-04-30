// SPDX-License-Identifier: Apache-2.0
/// Core engine module for Switchcraft.
///
/// Contains the storage protocol, search and indexing pipeline, residual codec,
/// k-means clustering, and tokenizer. Backend-agnostic: depends only on Swift
/// standard library plus Apple platform frameworks.
///
/// See `docs/Plan.md` for the full design.
public enum SwitchcraftCore {
    /// Semantic version of the package. Update with releases.
    public static let version = "0.0.0"
}
