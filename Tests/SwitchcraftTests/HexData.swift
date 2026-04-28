import Foundation

extension Data {
    /// Decode a lowercase hex string into bytes. Test-only helper.
    init(hexString: String) {
        var out = Data()
        out.reserveCapacity(hexString.count / 2)
        var index = hexString.startIndex
        while index < hexString.endIndex {
            let next = hexString.index(index, offsetBy: 2)
            if let byte = UInt8(hexString[index..<next], radix: 16) {
                out.append(byte)
            }
            index = next
        }
        self = out
    }
}
