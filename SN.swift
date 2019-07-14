import TensorFlow


public struct SNConv2D<Scalar: TensorFlowFloatingPoint>: Layer {

    // Avoid below errors:
    // TensorFlow.Layer:2:20: note: protocol requires nested type 'Input'; do you want to add it?
    // TensorFlow.Layer:3:20: note: protocol requires nested type 'Output'; do you want to add it?
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>

    // Copy of https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Layers/Convolutional.swift#L128
    public var filter: Tensor<Scalar>
    public var bias: Tensor<Scalar>
    @noDerivative public let activation: Activation
    @noDerivative public let strides: (Int, Int)
    @noDerivative public let padding: Padding
    @noDerivative public let dilations: (Int, Int)

    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>


    // Spectral Normalization parameters
    @noDerivative public let nPowerIteration: Int
    @noDerivative public let eps: Scalar
    @noDerivative public var u: Parameter<Scalar>
    @noDerivative public var v: Parameter<Scalar>


    // Copy of https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Layers/Convolutional.swift#L128
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1)
    ) {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dilations = dilations

        self.nPowerIteration = 1
        self.eps = 1e-12
        self.u = Parameter(Tensor<Scalar>(randomNormal: [filter.shape[3], 1]))
        self.v = Parameter(Tensor<Scalar>(randomNormal: [1, filter.shape[0..<3].contiguousSize]))
    }

    func normalize(_ x: Tensor<Scalar>, _ eps: Scalar) -> Tensor<Scalar> {
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
        let weightMatrix = filter.reshaped(to: [filter.shape[0..<3].contiguousSize, filter.shape[3]]).transposed()
        updateApproxVectors(nPowerIteration, weightMatrix)
        return activation(conv2D(
            input,
            filter: filter / calcMaxSingularValue(weightMatrix, u, v),
            strides: (1, strides.0, strides.1, 1),
            padding: padding) + bias)
    }

    @differentiable
    func applyingInference(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        let weightMatrix = filter.reshaped(to: [filter.shape[0..<3].contiguousSize, filter.shape[3]]).transposed()
        return activation(conv2D(
            input,
            filter: filter / calcMaxSingularValue(weightMatrix, u, v),
            strides: (1, strides.0, strides.1, 1),
            padding: padding) + bias)
    }

    @differentiable(vjp: _vjpApplied(to:))
    public func callAsFunction(_ input: Input) -> Output {
        switch Context.local.learningPhase {
        case .training: return applyingTraining(to: input)
        case .inference: return applyingInference(to: input)
        }
    }

    @usableFromInline
    func _vjpApplied(to input: Tensor<Scalar>) ->
        (Tensor<Scalar>, (Tensor<Scalar>) -> (SNConv2D<Scalar>.TangentVector, Tensor<Scalar>)) {
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
