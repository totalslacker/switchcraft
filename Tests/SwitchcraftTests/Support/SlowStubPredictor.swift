// SPDX-License-Identifier: Apache-2.0
import Foundation
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// A test-only `MLPredictor` stub that sleeps for a configurable duration
/// before returning valid (zeroed) embeddings.
///
/// Used by `T5CoreMLEmbedderConcurrencyTests` to verify that concurrent
/// encode calls are serialised by the actor's re-entrancy guard. Calling
/// code can inspect `callLog` after both encodes complete to assert ordering.
final class SlowStubPredictor: MLPredictor, @unchecked Sendable {

    let delay: Duration
    let dims: Int

    /// Each entry records the wall-clock start and end of one `predict` call.
    /// Written under `inFlight` actor serialisation; safe to read after awaiting
    /// both encodes without additional locking.
    private(set) var callLog: [(start: Date, end: Date)] = []

    init(delay: Duration, dims: Int = 128) {
        self.delay = delay
        self.dims = dims
    }

    func predict(input: any MLFeatureProvider) throws -> any MLFeatureProvider {
        let start = Date()

        let (seconds, attoseconds) = delay.components
        let interval = Double(seconds) + Double(attoseconds) / 1_000_000_000_000_000_000
        Thread.sleep(forTimeInterval: interval)

        let end = Date()
        callLog.append((start: start, end: end))

        // Determine windowSize from the input_ids shape ([1, windowSize]).
        guard let idsValue = input.featureValue(for: CoreMLInputName.inputIDs),
              let idsArray = idsValue.multiArrayValue else {
            fatalError("SlowStubPredictor: input missing '\(CoreMLInputName.inputIDs)'")
        }
        let windowSize = idsArray.shape[1].intValue

        // Return [1, windowSize, dims] Float32 tensors so the embedder's
        // readEmbeddingTensor / readRowL2Norms helpers succeed.
        let shape: [NSNumber] = [1, NSNumber(value: windowSize), NSNumber(value: dims)]

        // raw_projected: each element = 1.0 → per-row L2 norm ≈ sqrt(dims) >> 1.0
        let rawArray = try MLMultiArray(shape: shape, dataType: .float32)
        let rawPtr = rawArray.dataPointer.bindMemory(to: Float.self, capacity: rawArray.count)
        for i in 0..<rawArray.count { rawPtr[i] = 1.0 }

        // normalised: unit vectors (all 1/sqrt(dims))
        let normalisedArray = try MLMultiArray(shape: shape, dataType: .float32)
        let normPtr = normalisedArray.dataPointer.bindMemory(to: Float.self, capacity: normalisedArray.count)
        let invSqrtDims = 1.0 / Float(dims).squareRoot()
        for i in 0..<normalisedArray.count { normPtr[i] = invSqrtDims }

        return try MLDictionaryFeatureProvider(dictionary: [
            CoreMLOutputName.normalised:    MLFeatureValue(multiArray: normalisedArray),
            CoreMLOutputName.rawProjected:  MLFeatureValue(multiArray: rawArray),
        ])
    }
}

#endif
