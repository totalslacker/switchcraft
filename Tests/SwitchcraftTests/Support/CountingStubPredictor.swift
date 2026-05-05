// SPDX-License-Identifier: Apache-2.0
import Foundation
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// A fast test-only `MLPredictor` stub that returns valid zeroed embeddings
/// with no artificial delay.
///
/// Used by stress tests and reload-verification tests. Configure `failInterval`
/// to have the stub raise an IOSurface-like `NSException` periodically,
/// simulating ANE pool exhaustion without a real CoreML model.
final class CountingStubPredictor: MLPredictor, @unchecked Sendable {

    /// When non-nil, the stub raises an `NSException` on every call N where
    /// `N % failInterval == 0` (1-based counting). Use `failInterval: 1` to
    /// fail every call.
    let failInterval: Int?
    let dims: Int

    static let exceptionName   = "NSGenericException"
    static let exceptionReason = "E5 buffer object allocation failed"

    private var callCount: Int = 0

    init(failInterval: Int? = nil, dims: Int = 128) {
        self.failInterval = failInterval
        self.dims = dims
    }

    func predict(input: any MLFeatureProvider) throws -> any MLFeatureProvider {
        callCount += 1

        if let interval = failInterval, callCount % interval == 0 {
            NSException(
                name: NSExceptionName(Self.exceptionName),
                reason: Self.exceptionReason,
                userInfo: nil
            ).raise()
            fatalError("unreachable")
        }

        // Return valid [1, windowSize, dims] Float32 tensors.
        guard let idsValue = input.featureValue(for: CoreMLInputName.inputIDs),
              let idsArray = idsValue.multiArrayValue else {
            fatalError("CountingStubPredictor: input missing '\(CoreMLInputName.inputIDs)'")
        }
        let windowSize = idsArray.shape[1].intValue

        let shape: [NSNumber] = [1, NSNumber(value: windowSize), NSNumber(value: dims)]

        let rawArray = try MLMultiArray(shape: shape, dataType: .float32)
        let rawPtr = rawArray.dataPointer.bindMemory(to: Float.self, capacity: rawArray.count)
        for i in 0..<rawArray.count { rawPtr[i] = 1.0 }

        let normalisedArray = try MLMultiArray(shape: shape, dataType: .float32)
        let normPtr = normalisedArray.dataPointer.bindMemory(to: Float.self, capacity: normalisedArray.count)
        let invSqrtDims = 1.0 / Float(dims).squareRoot()
        for i in 0..<normalisedArray.count { normPtr[i] = invSqrtDims }

        return try MLDictionaryFeatureProvider(dictionary: [
            CoreMLOutputName.normalised:   MLFeatureValue(multiArray: normalisedArray),
            CoreMLOutputName.rawProjected: MLFeatureValue(multiArray: rawArray),
        ])
    }
}

/// A thread-safe counter for tracking how many times a predictor factory
/// closure is invoked. Pass an instance to a `@Sendable` factory closure;
/// call `increment()` inside the factory; read `count` in the test assertion.
///
/// Safe to use from actor-serialised factory closures without additional locking.
final class FactoryCallCounter: @unchecked Sendable {
    private var _count: Int = 0
    var count: Int { _count }
    func increment() { _count += 1 }
}

#endif
