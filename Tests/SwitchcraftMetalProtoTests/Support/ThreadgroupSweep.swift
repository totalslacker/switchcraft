// SPDX-License-Identifier: Apache-2.0
//
// Threadgroup tile shapes the prototype's sweep enumerates. The set was
// chosen by the Plan stage to cover (a) Apple Silicon's 32-wide SIMD-group
// boundary, (b) common "square" tile choices, and (c) one sub-simdgroup
// option (8 × 4) to characterise the lower edge.

import SwitchcraftMetalProto

enum ThreadgroupSweep {
    static let tiles: [MetalTile] = [
        MetalTile(8, 8),
        MetalTile(16, 16),
        MetalTile(32, 32),
        MetalTile(8, 32),
        MetalTile(32, 8),
        MetalTile(8, 4),
    ]
}
