public enum TokenizerError: Error, Sendable {
    case fileNotFound(path: String)
    case malformedJSON(underlying: any Error)
    case unsupportedComponent(String)
    case missingField(String)
    case unsupportedNormalizer(String)
}
