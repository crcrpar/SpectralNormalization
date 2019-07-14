import TensorFlow

public struct LayerParams: [Tensor<Scalar>] {
    public var data: [Tensor<Scalar>]
    public var weight: Tensor<Scalar>
    public var bias: Tensor<Scalar>?

    func init(_ buf: [Tensor<Scalar>], _ weightIdx: Int, biasIdx: Int?) {
        self.data = buf
        self.weight = buf[weightIdx]
        if biasIdx != nil {
            self.bias = buf[biasIdx!]
        }
    }
}

struct SpectralNormalization {
    var text = "Hello, World!"
}
