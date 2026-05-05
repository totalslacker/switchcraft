// SPDX-License-Identifier: Apache-2.0
import Foundation
@testable import SwitchcraftCoreML

#if canImport(CoreML)
import CoreML

/// A test-only `MLPredictor` stub that raises an Objective-C exception
/// from inside `predict(input:)`, simulating CoreML internal failure sites
/// such as `MLE5BindEmptyMemoryObjectToPort`.
///
/// Used by `T5CoreMLEmbedderFailureLogTests` to verify that:
/// - `catchingNSException` converts the ObjC exception into a
///   `CoreMLNativeError.nativeException` thrown value (no abort).
/// - The embedder writes a valid JSONL row to `failureLogURL`.
final class ThrowingStubPredictor: MLPredictor, Sendable {

    static let exceptionName   = "TestCrash"
    static let exceptionReason = "deliberate test exception"

    func predict(input: any MLFeatureProvider) throws -> any MLFeatureProvider {
        NSException(
            name: NSExceptionName(Self.exceptionName),
            reason: Self.exceptionReason,
            userInfo: nil
        ).raise()
        // Unreachable; `raise()` always throws an ObjC exception.
        fatalError("unreachable")
    }
}

#endif
