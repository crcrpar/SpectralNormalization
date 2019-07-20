import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(S4TFSpectralNormalizationTests.allTests),
    ]
}
#endif
