// SPDX-License-Identifier: Apache-2.0
import Foundation

#if canImport(CoreML)
import CoreML

/// Testability seam for `MLModel.prediction(from:)`.
///
/// `T5CoreMLEmbedder` stores `any MLPredictor` rather than `MLModel` directly,
/// allowing tests to inject stub predictors (e.g., `SlowStubPredictor`,
/// `ThrowingStubPredictor`) via the internal init without loading a real
/// `.mlpackage` asset. Production code uses `MLModel` via the extension below.
///
/// `internal` to `SwitchcraftCoreML`; access from tests via `@testable import`.
internal protocol MLPredictor: Sendable {
    func predict(input: any MLFeatureProvider) throws -> any MLFeatureProvider
}

// @unchecked @retroactive: retroactive conformances on CoreML types; safe
// because all access is gated through T5CoreMLEmbedder's actor isolation.
extension MLModel: @unchecked @retroactive Sendable {}
extension MLDictionaryFeatureProvider: @unchecked @retroactive Sendable {}

extension MLModel: MLPredictor {
    internal func predict(input: any MLFeatureProvider) throws -> any MLFeatureProvider {
        try self.prediction(from: input)
    }
}

#endif
