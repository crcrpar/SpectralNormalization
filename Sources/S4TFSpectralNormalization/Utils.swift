import TensorFlow


/// Normalizes input vector with its L2 norm.
///
/// - Parameters:
///   - x: A 1-D tensor or 2-D tensor with its one axis is sized 1.
///   - eps: A value for numerical stability to avoid zero division.
public func normalize(_ x: Tensor<Scalar>, _ eps: Tensor<Scalar>) -> Tensor<Scalar> {
    return x / (sqrt(x.squared().sum()) + eps)
}


/// Updates the approximate singular vectors with the current weight (= `filter`).
///
/// - Parameters:
///   - nPowerIteration: The number of iterations to update two vectors.
///   - weightMatrix: The 2-D tensor obtained by reshaping the latest weight (= `filter`) into 2-D.
///     For `Conv2D`, the shape is [output channel count, (filter height) * (filter width) * (input channel count)].
///     For 2-D `Dense`, the shape is [output size, input size], which is equivalent to transposed weight.
public func updateApproxVectors(_ nPowerIteration: Int, _ weightMatrix: Tensor<Scalar>) {
    for _ in 0..<nPowerIteration {
        v.value = normalize(u.value.transposed() • weightMatrix, eps)
        u.value = normalize(weightMatrix • v.value.transposed(), eps)
    }
}


/// Calculates the approximate maximum singular value of weight (= `filter`) from its 2-D version.
///
/// - Parameters
///   - weightMatrix: `filter` that is reshaped into 2-D.
///   - u: The approximate left singular vector.
///   - v: The approximate right singular vector.
@differentiable
public func calcMaxSingularValue(_ weightMatrix: Tensor<Scalar>, _ u: Parameter<Scalar>, _ v: Parameter<Scalar>) -> Tensor<Scalar> {
    let sigma = u.value.transposed() • weightMatrix • v.value.transposed()
    return sigma
}
