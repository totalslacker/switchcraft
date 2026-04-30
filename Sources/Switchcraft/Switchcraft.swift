// SPDX-License-Identifier: Apache-2.0
@_exported import SwitchcraftCore

/// Public umbrella module for Switchcraft.
///
/// Re-exports `SwitchcraftCore` and will host the public `SwitchcraftStore`
/// actor API. Library consumers normally `import Switchcraft`.
///
/// See `docs/Plan.md` for the full design.
public enum Switchcraft {
    /// Semantic version of the package. Update with releases.
    public static let version = SwitchcraftCore.version
}
