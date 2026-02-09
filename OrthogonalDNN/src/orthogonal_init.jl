
# orthogonal_init.jl – Orthogonal matrix generation for weight initialisation

"""
    orthogonal_matrix(rows::Int, cols::Int; rng=Random.default_rng(), T=Float32)

Generate a (semi-)orthogonal matrix of size `rows × cols` using the Householder
QR factorisation implemented in this package.

When `rows ≥ cols` the returned matrix has orthonormal *columns*.
When `rows < cols` the returned matrix has orthonormal *rows*.

# Keyword Arguments
- `rng`: Random-number generator (default: global RNG).
- `T::Type`: Element type (default `Float32`, appropriate for Flux).

# Returns
- `Q::Matrix{T}`: A `rows × cols` (semi-)orthogonal matrix.
"""
function orthogonal_matrix(rows::Int, cols::Int;
                           rng::AbstractRNG = Random.default_rng(),
                           T::Type{<:AbstractFloat} = Float32)
    if rows ≥ cols
        A = randn(rng, T, rows, cols) .* √(T(2))
        Y, W, _ = wy_householder(A)
        Q = I + W * Y'
        return Matrix{T}(Q[:, 1:cols])
    else
        A = randn(rng, T, cols, rows) .* √(T(2))
        Y, W, _ = wy_householder(A)
        Q = I + W * Y'
        Q = Q[:, 1:rows]
        return Matrix{T}(Q')
    end
end

"""
    orthogonal_init(rows::Int, cols::Int; kwargs...)

Flux-compatible initialiser that returns an orthogonal matrix.
Can be passed directly as the `init` keyword to any Flux layer.

# Example
```julia
Dense(128 => 64; init = orthogonal_init)
```
"""
function orthogonal_init(rows::Int, cols::Int; kwargs...)
    orthogonal_matrix(rows, cols; kwargs...)
end

"""
    orthogonal_init(rng::AbstractRNG, rows::Int, cols::Int; kwargs...)

Variant accepting an explicit RNG, matching the Flux initialiser signature
`init(rng, dims...)`.
"""
function orthogonal_init(rng::AbstractRNG, rows::Int, cols::Int; kwargs...)
    orthogonal_matrix(rows, cols; rng = rng, kwargs...)
end
