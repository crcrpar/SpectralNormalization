import XCTest

import TensorFlow
@testable import SpectralNormalization


// Constants used in assertion.
let zero = Tensor<Float>(0)
let eps = Tensor<Float>(1e-12)


final class SpectralNormalizationTests: XCTestCase {
    
    func testSNConv2DTraining() {
        let (batchSize, cIn, cOut, h, w) = (10, 3, 16, 8, 8)
        let filter = Tensor<Float>(randomNormal: TensorShape([h, w, cIn, cOut]))
        let bias = Tensor<Float>(zeros: TensorShape([cOut]))
        let snconv2d = SNConv2D<Float>(filter: filter, bias: bias, padding: .same)
        
        let inputs1 = Tensor<Float>(randomNormal: TensorShape([batchSize, h, w, cIn]))
        
        Context.local.learningPhase = .training
        let (u1, v1) = (snconv2d.u.value, snconv2d.v.value)
        let _ = snconv2d(inputs1)
        let (u2, v2) = (snconv2d.u.value, snconv2d.v.value)
        
        XCTAssertTrue((u2 - u1).mean() != zero)
        XCTAssertTrue((v2 - v1).mean() != zero)
    }
    
    func testSNConv2DInference() {
        let (batchSize, cIn, cOut, h, w) = (10, 3, 16, 8, 8)
        let filter = Tensor<Float>(randomNormal: TensorShape([h, w, cIn, cOut]))
        let bias = Tensor<Float>(zeros: TensorShape([cOut]))
        let snconv2d = SNConv2D<Float>(filter: filter, bias: bias, padding: .same)
        
        let inputs1 = Tensor<Float>(randomNormal: TensorShape([batchSize, h, w, cIn]))
        
        Context.local.learningPhase = .inference
        let (u1, v1) = (snconv2d.u.value, snconv2d.v.value)
        let _ = snconv2d(inputs1)
        let (u2, v2) = (snconv2d.u.value, snconv2d.v.value)
        

        XCTAssertTrue(abs((u2 - u1).mean()) <= eps)
        XCTAssertTrue(abs((v2 - v1).mean()) <= eps)
    }
    
    func testSNDenseTraining() {
        let (batchSize, inSize, outSize) = (5, 10, 15)
        let weight = Tensor<Float>(randomNormal: TensorShape([inSize, outSize]))
        let bias = Tensor<Float>(zeros: [outSize])
        let sndense = SNDense<Float>(weight: weight, bias: bias)
        
        let inputs1 = Tensor<Float>(randomNormal: TensorShape([batchSize, inSize]))
        
        Context.local.learningPhase = .training
        let (u1, v1) = (sndense.u.value, sndense.v.value)
        let _ = sndense(inputs1)
        let (u2, v2) = (sndense.u.value, sndense.v.value)
        
        XCTAssertTrue((u2 - u1).mean() != zero)
        XCTAssertTrue((v2 - v1).mean() != zero)
    }
    
    func testSNDenseInference() {
        let (batchSize, inSize, outSize) = (5, 10, 15)
        let weight = Tensor<Float>(randomNormal: TensorShape([inSize, outSize]))
        let bias = Tensor<Float>(zeros: [outSize])
        let sndense = SNDense<Float>(weight: weight, bias: bias)
        
        let inputs1 = Tensor<Float>(randomNormal: TensorShape([batchSize, inSize]))
        
        Context.local.learningPhase = .inference
        let (u1, v1) = (sndense.u.value, sndense.v.value)
        let _ = sndense(inputs1)
        let (u2, v2) = (sndense.u.value, sndense.v.value)
        
        XCTAssertTrue(abs((u2 - u1).mean()) <= eps)
        XCTAssertTrue(abs((v2 - v1).mean()) <= eps)
    }

    static var allTests = [
        ("testSNConv2DTraining", testSNConv2DTraining),
        ("testSNConv2DInference", testSNConv2DInference),
        ("testSNDenseTraining", testSNDenseTraining),
        ("testSNDenseInference", testSNDenseInference),
    ]
}
