import Foundation

extension Data {
    /// Decode a lowercase hex string into bytes. Test-only helper.
    /// Traps on odd-length input or any pair that isn't valid hex so
    /// fixture corruption surfaces immediately.
    init(hexString: String) {
        precondition(
            hexString.count.isMultiple(of: 2),
            "Hex string must contain an even number of characters."
        )
        var out = Data()
        out.reserveCapacity(hexString.count / 2)
        var index = hexString.startIndex
        while index < hexString.endIndex {
            guard let next = hexString.index(
                index, offsetBy: 2, limitedBy: hexString.endIndex
            ) else {
                preconditionFailure("Hex string ended unexpectedly while decoding.")
            }
            let pair = hexString[index..<next]
            guard let byte = UInt8(pair, radix: 16) else {
                preconditionFailure("Invalid hex byte pair: \(pair)")
            }
            out.append(byte)
            index = next
        }
        self = out
    }
}
