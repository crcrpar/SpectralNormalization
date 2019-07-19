import TensorFlow


public struct SNLinear<Scalar: TensorFlowFloatingPoint>: Layer {

    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>

    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    @noDerivative public let activation: Activation
    // @noDerivative internal let batched: Bool  // Deliberately removed for simplicity

    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    // SN parameters
    @noDerivative public var nPowerIteration: Int
    @noDerivative public var eps: Tensor<Scalar>
    @noDerivative public var u: Parameter<Scalar>
    @noDerivative public var v: Parameter<Scalar>

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation,
        nPowerIteration: Int = 1,
        eps: Scalar = 1e-12
    ) {
        precondition(weight.rank == 2, "The rank of the 'weight' tensor must be 2.")
        // precondition(bias.rank <= 2, "The rank of the 'bias' tensor must be less than 3.")
        self.weight = weight
        self.bias = bias
        self.activation = activation
        // self.batched = weight.rank == 3

        self.nPowerIteration = nPowerIteration
        self.eps = Tensor<Scalar>(eps)
        let weightShape = weight.shape
        self.u = Parameter(normalize(Tensor<Scalar>(randomNormal: [filter.shape[1]]), eps))
        self.v = Parameter(Tensor<Scalar>(zeros: [filter.shape[0]]))
    }

    /// Normalizes input vector with its L2 norm.
    func normalize(_ x: Tensor<Scalar>, _ eps: Tensor<Scalar>) -> Tensor<Scalar> {
        return x / (sqrt(x.squared().sum()) + eps)
    }

    func updateApproxVectors(_ nPowerIteration: Int, _ weightMatrix: Tensor<Scalar>) {
        for _ in 0..<nPowerIteration {
            v.value = normalize(u.value.transposed() • weightMatrix, eps)
            u.value = normalize(weightMatrix • v.value.transposed(), eps)
        }
    }

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

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        swift Context.local.learningPhase {
            case .training: return applyingTraining(to: input)
            case .inference: return applyingInference(to: input)
        }
    }
}
