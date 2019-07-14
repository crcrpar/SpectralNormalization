import TensorFlow

let nPowerIteration = 1
let eps: Float = 1e-12

func normalize(_ x: Tensor<Float>, _ eps: Float) -> Tensor<Float> {
    return x / (sqrt(x.squared().sum()) + eps)
}

var conv2d = Conv2D<Float>(filterShape: (3, 3, 1, 4))
print(conv2d.filter.shape)

var weight = conv2d.filter
var weightMatrix = weight.reshaped(to: [weight.shape[0..<3].contiguousSize, weight.shape[3]])
weightMatrix = weightMatrix.transposed()

print("weightMatrix.shape: \(weightMatrix.shape)")

var u = Tensor<Float>(randomNormal: [weightMatrix.shape[0], 1])
var v = Tensor<Float>(randomNormal: [1, weightMatrix.shape[1]])
print("u.shape: \(u.shape), v.shape: \(v.shape)")

for _ in 0..<nPowerIteration {
    v = normalize(u.transposed() • weightMatrix, eps)
    u = normalize(weightMatrix • v.transposed(), eps)
}

print("u.shape: \(u.shape), v.shape: \(v.shape)")

let sigma = u.transposed() • weightMatrix • v.transposed()
print(sigma.shape)

weight = weight / sigma


func reshapeConv2DWeight<Scalar>(_ layer: Conv2D<Scalar>) -> Tensor<Scalar> {
    // Move the axis of out_channels to the first and flatten
    return layer.filter.reshaped(to: [weight.shape[0..<3].contiguousSize, weight.shape[3]]).transposed()
}

func updateApproxVectors(_ nPowerIteration: Int, _ weight: Tensor<Float>, _ _u: Tensor<Float>, _ _v: Tensor<Float>) -> (Tensor<Float>, Tensor<Float>)  {
    var u = _u
    var v = _v
    for _ in 0..<nPowerIteration {
        v = normalize(u.transposed() • weightMatrix, eps)
        u = normalize(weightMatrix • v.transposed(), eps)
    }
    return (u, v)
}

func updateApproxVectors(_ nPowerIteration: Int, _ weight: Tensor<Float>, _ u: inout Tensor<Float>, _ v: inout Tensor<Float>) -> (Tensor<Float>, Tensor<Float>) {
    for _ in 0..<nPowerIteration {
        v = normalize(u.transposed() • weightMatrix, eps)
        u = normalize(weightMatrix • v.transposed(), eps)
    }
    return (u, v)
}

func calcMaxSingularValue(_ weightMatrix: Tensor<Float>,
                          _ u: inout Tensor<Float>,
                          _ v: inout Tensor<Float>) -> Tensor<Float> {
    assert(weightMatrix.shape.count == 2)
    for _ in 0..<nPowerIteration {
        v = normalize(u.transposed() • weightMatrix, eps)
        u = normalize(weightMatrix • v.transposed(), eps)
    }
    let sigma = u.transposed() • weightMatrix • v.transposed()
    return sigma
}


var (u1, v1) = (u, v)

print("u: \(u1)")
print("v: \(v1)")
let sigma2 = calcMaxSingularValue(weightMatrix, &u1, &v1)
print("u: \(u1)")
print("v: \(v1)")


for _ in 0..<nPowerIteration {
    v = normalize(u.transposed() • weightMatrix, eps)
    u = normalize(weightMatrix • v.transposed(), eps)
}

print("u.shape: \(u.shape), v.shape: \(v.shape)")

let sigma3 = u.transposed() • weightMatrix • v.transposed()
print(sigma3.shape)

weight = weight / sigma3
