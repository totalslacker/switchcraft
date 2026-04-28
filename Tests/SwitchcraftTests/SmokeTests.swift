import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

@Suite("Scaffolding Smoke Tests")
struct SmokeTests {

    @Test("All modules import and expose a version string")
    func modulesImport() {
        #expect(Switchcraft.version == SwitchcraftCore.version)
        #expect(SwitchcraftSQLite.version == SwitchcraftCore.version)
        #expect(!SwitchcraftCore.version.isEmpty)
    }
}
