// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCoreMLObjC

/// A caught Objective-C exception from a CoreML prediction call.
///
/// CoreML's internal failure sites (`MLE5BindEmptyMemoryObjectToPort` and
/// similar) raise NSException rather than returning NSError. Swift's do-catch
/// cannot intercept ObjC exceptions — they propagate past Swift's error
/// machinery into std::terminate. This error type carries the exception
/// through as a first-class Swift error so callers can handle it gracefully.
///
/// Both this type and `catchingNSException` are `internal` to SwitchcraftCoreML.
/// Access from tests via `@testable import SwitchcraftCoreML`.
internal enum CoreMLNativeError: Error, Sendable {
    /// An Objective-C exception raised during a CoreML prediction call.
    /// - Parameters:
    ///   - name: `NSException.name.rawValue`
    ///   - reason: `NSException.reason` (empty string if nil)
    ///   - callStack: first 5 frames of `NSException.callStackSymbols`
    case nativeException(name: String, reason: String, callStack: [String])
}

/// Execute `body` inside an Objective-C @try/@catch guard.
///
/// ObjC exceptions raised by `body` (e.g., from CoreML internal failure sites)
/// are caught by the ObjC bridge and translated into `CoreMLNativeError.nativeException`
/// rather than propagating to std::terminate. Swift errors thrown by `body`
/// are captured inside the ObjC block and rethrown normally after the bridge
/// returns; they are unaffected by the ObjC exception mechanism.
///
/// `NSException` fields are extracted synchronously inside this function; no
/// ObjC reference escapes the call boundary, satisfying Swift 6 Sendability.
internal func catchingNSException<T>(_ body: () throws -> T) throws -> T {
    var result: T? = nil
    var swiftError: (any Error)? = nil

    // The ObjC block is void-returning and non-escaping. Swift errors thrown
    // by body() are captured here; ObjC exceptions bypass Swift's catch and
    // propagate directly to the ObjC @try/@catch layer.
    let caught = MLExceptionCatcher.perform {
        do {
            result = try body()
        } catch {
            swiftError = error
        }
    }

    if let exception = caught {
        // Extract all ObjC fields synchronously before this function returns.
        let name = exception.name.rawValue
        let reason = exception.reason ?? ""
        let callStack = Array(exception.callStackSymbols.prefix(5))
        throw CoreMLNativeError.nativeException(name: name, reason: reason, callStack: callStack)
    }

    if let error = swiftError {
        throw error
    }

    // result is always non-nil if neither ObjC exception nor Swift error occurred.
    return result!
}
