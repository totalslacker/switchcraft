// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import SwitchcraftCoreML

/// Unit tests for the `catchingNSException` bridge and `CoreMLNativeError`.
///
/// These tests do NOT require any CoreML model asset and run unconditionally.
@Suite("MLExceptionCatcher bridge")
struct MLExceptionCatcherTests {

    // MARK: - Exception capture

    @Test("catchingNSException converts ObjC exception to CoreMLNativeError")
    func catchesObjCException() throws {
        let result = Result<Int, any Error> {
            try catchingNSException {
                NSException(
                    name: NSExceptionName("BridgeTestException"),
                    reason: "test reason",
                    userInfo: nil
                ).raise()
                return 0  // unreachable
            }
        }

        switch result {
        case .success:
            Issue.record("Expected CoreMLNativeError to be thrown, but block returned normally")
        case .failure(let error):
            guard case CoreMLNativeError.nativeException(
                let name, let reason, let callStack
            ) = error else {
                Issue.record("Expected CoreMLNativeError.nativeException, got \(error)")
                return
            }
            #expect(name == "BridgeTestException")
            #expect(reason == "test reason")
            // callStackSymbols is populated by the ObjC runtime at raise() time.
            #expect(!callStack.isEmpty, "callStack should contain at least one frame")
        }
    }

    // MARK: - Pass-through

    @Test("catchingNSException returns value when no exception is raised")
    func passThrough() throws {
        let value = try catchingNSException { 42 }
        #expect(value == 42)
    }

    @Test("catchingNSException propagates Swift errors without conversion")
    func propagatesSwiftErrors() {
        struct TestError: Error {}
        let result = Result<Int, any Error> {
            try catchingNSException { throw TestError() }
        }
        switch result {
        case .success:
            Issue.record("Expected TestError to be thrown")
        case .failure(let error):
            #expect(error is TestError)
        }
    }
}
