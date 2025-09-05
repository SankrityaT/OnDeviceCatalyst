import XCTest
@testable import OnDeviceCatalyst

final class OnDeviceCatalystTests: XCTestCase {
    
    func testModelProfileCreation() throws {
        // Test creating a model profile with a mock path
        let mockPath = "/mock/path/test-model.gguf"
        
        // This will fail file validation but we can test the structure
        XCTAssertThrowsError(try ModelProfile(filePath: mockPath)) { error in
            XCTAssertTrue(error is CatalystError)
        }
    }
    
    func testInstanceSettingsValidation() throws {
        let settings = InstanceSettings.iphone16ProMax
        
        // Test that our optimized settings are valid
        XCTAssertNoThrow(try settings.validate())
        
        // Verify the optimized values
        XCTAssertEqual(settings.contextLength, 2048)
        XCTAssertEqual(settings.batchSize, 256)
        XCTAssertEqual(settings.gpuLayers, 25)
        XCTAssertEqual(settings.cpuThreads, 6)
    }
    
    func testPredictionConfigPresets() {
        let speedConfig = PredictionConfig.speed
        let qualityConfig = PredictionConfig.quality
        
        // Speed should prioritize performance
        XCTAssertLessThan(speedConfig.temperature, qualityConfig.temperature)
        
        // Quality should have more tokens
        XCTAssertGreaterThan(qualityConfig.maxTokens, speedConfig.maxTokens)
    }
    
    func testCatalystServiceInitialization() {
        let catalyst = Catalyst.shared
        XCTAssertNotNil(catalyst)
    }
}
