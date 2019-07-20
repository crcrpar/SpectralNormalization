import TensorFlow


/// A densely-connected neural network layer with spectrally normalized weight matrix.
///
/// `SNDense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a spectrally normalized weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
///
/// This layer differs from `Dense` in that this does not support 3-D weight tensors with 2-D bias matrices
/// for ease of implementation.
public struct SNDense<Scalar: TensorFlowFloatingPoint>: Layer {

    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>

    /// The weight matrix.
    public var weight: Tensor<Scalar>
    /// The bias vector
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    // NOTE (crcrpar): Deliberately removed `batched: Bool` for simplicity

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    
    // Spectral Normalization parameters
    // TODO (crcpar): Support nPowerIteration > 1.
    /// The number of iterations to update approximate left and right singular vectors.
    @noDerivative public var nPowerIteration: Int
    // TODO (crcpar): Make `eps` a `Scalar` after `TF-625`
    /// The epsilon value to avoid zero-division in vectors' L2 norm normalization.
    @noDerivative public var eps: Tensor<Scalar>
    /// The approximate left singular vector.
    @noDerivative public var u: Parameter<Scalar>
    /// The approximate right singular vector.
    @noDerivative public var v: Parameter<Scalar>

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        nPowerIteration: Int = 1,
        eps: Scalar = 1e-12
    ) {
        precondition(weight.rank == 2, "The rank of the 'weight' tensor must be 2 for SNDense.")
        precondition(bias.rank == 1, "The rank of the 'bias' tensor must be 1 for SNDense.")
        self.weight = weight
        self.bias = bias
        self.activation = activation

        self.nPowerIteration = nPowerIteration
        self.eps = Tensor<Scalar>(eps)
        let weightShape = weight.shape
        self.u = Parameter(Tensor<Scalar>(randomNormal: [weightShape[1], 1]))
        self.v = Parameter(Tensor<Scalar>(zeros: [1, weightShape[0]]))
    }

    /// Normalizes input vector with its L2 norm.
    func normalize(_ x: Tensor<Scalar>, _ eps: Tensor<Scalar>) -> Tensor<Scalar> {
        return x / (sqrt(x.squared().sum()) + eps)
    }

    /// Updates the approximate singular vectors with the current weight (= `filter`).
    ///
    /// - Parameters:
    ///   - nPowerIteration: The number of iterations to update two vectors.
    ///   - weightMatrix: The 2-D tensor obtained by reshaping the latest weight (= `filter`) into 2-D.
    ///     The shape is [output channel count, (filter height) * (filter width) * (input channel count)].
    func updateApproxVectors(_ nPowerIteration: Int, _ weightMatrix: Tensor<Scalar>) {
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
    func calcMaxSingularValue(_ weightMatrix: Tensor<Scalar>, _ u: Parameter<Scalar>, _ v: Parameter<Scalar>) -> Tensor<Scalar> {
        let sigma = u.value.transposed() • weightMatrix • v.value.transposed()
        return sigma
    }

    @differentiable
    func applyingTraining(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let weightMatrix = weight.transposed()
        updateApproxVectors(nPowerIteration, weightMatrix)
        return activation(matmul(
            input,
            weight / calcMaxSingularValue(weightMatrix, u, v)) + bias)
    }

    @differentiable
    func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let weightMatrix = weight.transposed()
        return activation(matmul(
            input,
            weight / calcMaxSingularValue(weightMatrix, u, v)) + bias)
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The ouptut.
    @differentiable(vjp: _vjpApplied(to:))
    public func callAsFunction(_ input: Input) -> Output {
        switch Context.local.learningPhase {
            case .training: return applyingTraining(to: input)
            case .inference: return applyingInference(to: input)
        }
    }
    
    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (SNDense<Scalar>.TangentVector, Tensor<Scalar>)) {
        switch Context.local.learningPhase {
        case .training:
            return valueWithPullback(at: input) {
                $0.applyingTraining(to: $1)
            }
        case .inference:
            return valueWithPullback(at: input) {
                $0.applyingInference(to: $1)
            }
        }
    }
}
