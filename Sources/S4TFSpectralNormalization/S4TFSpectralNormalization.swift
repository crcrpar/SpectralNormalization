import Foundation

import TensorFlow


/// Spectrally Normalized 2-D Convolution Layer.
///
/// This layer creates a convolution filter that is convolved with the layer input to produce
/// a tensor of outputs after the filter is normalized by its maximum singular value which is
/// computed via power iteration method.
public struct SNConv2D<Scalar: TensorFlowFloatingPoint>: Layer {

    // TODO (crcrpar): Consider removing these two `typealias`s.
    // Avoid below errors:
    // TensorFlow.Layer:2:20: note: protocol requires nested type 'Input'; do you want to add it?
    // TensorFlow.Layer:3:20: note: protocol requires nested type 'Output'; do you want to add it?
    public typealias Input = Tensor<Scalar>
    public typealias Output = Tensor<Scalar>

    // Copy of https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Layers/Convolutional.swift#L128
    /// The 4-D convolution filter.
    public var filter: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: (Int, Int)
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for spatial dimensions.
    @noDerivative public let dilations: (Int, Int)

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

    // The base initializer
    // This initializer currently does not support `nPowerIteration` and `eps`
    // Copy of https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Layers/Convolutional.swift#L128
    /// Creates a `SNConv2D` layer with specified filter, bias, activation function, strides,
    /// dilations and padding.
    ///
    /// - Parameters:
    ///   - filter: The 4-D convolution filter of shape
    ///     [filter height, filter width, input channel count, output channel count]
    ///   - bias: The bias vector of shape [output channel count].
    ///   - activation: The element-wise activation function.
    ///   - strides: The strides of the sliding window for spatial dimensions, i.e.
    ///     (stride height, stride width).
    ///   - padding: The padding algorithm for convolution.
    ///   - dilations: The dilation factors for spatial dimensions, i.e.
    ///     (dilation height, dilation width).
    public init(
        filter: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation = identity,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid,
        dilations: (Int, Int) = (1, 1),
        nPowerIteration: Int = 1,
        eps: Scalar = 1e-12) {

        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dilations = dilations

        self.nPowerIteration = nPowerIteration
        self.eps = Tensor<Scalar>(eps)
        self.u = Parameter(Tensor<Scalar>(randomNormal: [filter.shape[3], 1]))
        self.v = Parameter(Tensor<Scalar>(randomNormal: [1, filter.shape[0..<3].contiguousSize]))
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
