import Testing
import SwitchcraftCore
import SwitchcraftStorageTesting

@Suite("InMemoryStorage Conformance")
struct InMemoryStorageConformanceTests {

    @Test("InMemoryStorage satisfies the SwitchcraftStorage contract")
    func conformance() async throws {
        try await StorageConformance.runAll {
            InMemoryStorage()
        }
    }
}
