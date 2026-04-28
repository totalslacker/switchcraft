public enum TokenizerError: Error, @unchecked Sendable {
    case fileNotFound(path: String)
    case ioError(path: String, underlying: any Error)
    case malformedJSON(underlying: any Error)
    case unsupportedComponent(String)
    case missingField(String)
    case unsupportedNormalizer(String)
}

// `malformedJSON` and `ioError` carry an `any Error` payload that is not
// guaranteed to be Sendable (e.g. NSError from JSONSerialization). We mark
// the enum `@unchecked Sendable` because the underlying error is only
// inspected by the catching thread for diagnostic purposes; it is not
// shared mutable state. Callers that need to ferry the underlying error
// across actor boundaries should extract a String description first.
