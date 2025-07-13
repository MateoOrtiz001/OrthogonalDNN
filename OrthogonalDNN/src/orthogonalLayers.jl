"""
    orthogonalDense(inDim, outDim, activation=Flux.relu, seed=nothing)

Returns a `Dense` layer with the weights matrix initialized orthogonally.

# Arguments
- `inDim::Int`: Number of neurons of the previous layer.
- `outDim::Int`: Number of neurons of the current layer.
- `activation`: Activation function of the layer. Defaults to `Flux.relu`.
- `seed::Int`: Optional seed for random initialization. Defaults to `nothing`.

# Output
- `layer`: A `Dense` layer with orthogonal weights `Q`, zero bias `b`, and given activation.

# Example
```julia
h = orthogonalDense(784, 256, Flux.sigmoid, 43)
h = orthogonalDense(128, 128)
"""
function orthogonalDense(inDim::Int, outDim::Int, activation = Flex.relu, seed = nothing)
    Q = orthogonal_matrix(outDim,inDim,seed)
    b = zeros(outDim)
    layer = Dense(Q,b,activation)
    return layer
end