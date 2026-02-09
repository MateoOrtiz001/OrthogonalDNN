# ──────────────────────────────────────────────────────────────────────────────
# layers.jl – Flux-compatible orthogonal neural-network layers
#
# Three layer types are provided:
#
#   • OrthogonalDense   – Standard Dense layer *initialised* with orthogonal
#                         weights. Training is unconstrained (weights may drift
#                         from orthogonality; pair with a regulariser).
#
#   • CayleyDense       – Fully orthogonal (square) layer using the Cayley
#                         transform of a learnable skew-symmetric matrix.
#                         Weights stay exactly orthogonal throughout training.
#
#   • HouseholderLayer  – Fully orthogonal layer parametrised as a product of
#                         Householder reflections. Supports non-square
#                         (m ≥ n) transformations.
# ──────────────────────────────────────────────────────────────────────────────

#  OrthogonalDense

"""
    OrthogonalDense(in => out, σ = identity; bias = true, T = Float32)

A standard `Dense` layer whose weight matrix is *initialised* to be
(semi-)orthogonal via the Householder QR factorisation.

During training the weights are updated freely, so they may lose
orthogonality.  Combine with [`orthogonal_regularizer`](@ref) to keep
them close to orthogonal.

# Arguments
- `in::Int`:  Input  dimension.
- `out::Int`: Output dimension.
- `σ`:        Activation function (default `identity`).

# Keyword Arguments
- `bias::Bool`: Include a bias vector (default `true`).
- `T::Type`:    Element type (default `Float32`).

# Example
```julia
layer = OrthogonalDense(784 => 256, relu)
y = layer(randn(Float32, 784, 32))   # batch of 32
```
"""
struct OrthogonalDense{F, M <: AbstractMatrix, B}
    weight::M
    bias::B
    σ::F
end

Flux.@layer OrthogonalDense

function OrthogonalDense((in_dim, out_dim)::Pair{Int,Int}, σ = identity;
                         bias::Bool = true, T::Type{<:AbstractFloat} = Float32)
    W = orthogonal_matrix(out_dim, in_dim; T = T)
    b = bias ? zeros(T, out_dim) : Flux.Zeros()
    return OrthogonalDense(W, b, σ)
end

function (m::OrthogonalDense)(x::AbstractVecOrMat)
    return m.σ.(m.weight * x .+ m.bias)
end

function (m::OrthogonalDense)(x::AbstractArray)
    # Reshape for batched inputs (e.g. 3D+ arrays)
    sz = size(x)
    x2 = reshape(x, sz[1], :)
    y  = m(x2)
    return reshape(y, size(y, 1), sz[2:end]...)
end

function Base.show(io::IO, m::OrthogonalDense)
    in_dim  = size(m.weight, 2)
    out_dim = size(m.weight, 1)
    σ_name  = m.σ === identity ? "" : ", $(m.σ)"
    print(io, "OrthogonalDense($(in_dim) => $(out_dim)$(σ_name))")
end


#  CayleyDense – always orthogonal via the Cayley transform

"""
    CayleyDense(n => n, σ = identity; bias = true, T = Float32)

A *fully* orthogonal dense layer for **square** transformations.

The weight matrix is parametrised as

    W = (I − A)(I + A)⁻¹

where `A` is a learnable *skew-symmetric* matrix (Aᵀ = −A).  The Cayley
transform guarantees that `W` is orthogonal for every value of `A`, so
no regulariser is needed.

# Constraints
Input and output dimensions must be equal (`n => n`).

# Arguments
- `n::Int`: Dimension (both input and output).
- `σ`:      Activation function (default `identity`).

# Keyword Arguments
- `bias::Bool`: Include a bias vector (default `true`).
- `T::Type`:    Element type (default `Float32`).

# Example
```julia
layer = CayleyDense(128 => 128, relu)
y = layer(randn(Float32, 128, 64))
```
"""
struct CayleyDense{F, M <: AbstractMatrix, B}
    A::M        # learnable upper-triangular entries (skew-symmetric is derived)
    bias::B
    σ::F
    n::Int
end

Flux.@layer CayleyDense

function CayleyDense((in_dim, out_dim)::Pair{Int,Int}, σ = identity;
                     bias::Bool = true, T::Type{<:AbstractFloat} = Float32)
    in_dim == out_dim || throw(DimensionMismatch(
        "CayleyDense requires square transformation (got $in_dim => $out_dim). " *
        "Use OrthogonalDense or HouseholderLayer for non-square."))
    n = in_dim
    # Initialise A as small random skew-symmetric (store upper triangle)
    A = randn(T, n, n) .* T(0.01)
    b = bias ? zeros(T, n) : Flux.Zeros()
    return CayleyDense(A, b, σ, n)
end

"""
    cayley_weight(m::CayleyDense)

Materialise the orthogonal weight matrix W = (I − A)(I + A)⁻¹
from the learnable parameter `A`.
"""
function cayley_weight(m::CayleyDense)
    A_skew = m.A - m.A'  # ensure skew-symmetry
    A_skew = A_skew .* eltype(m.A)(0.5)
    Id = Matrix{eltype(m.A)}(I, m.n, m.n)
    W = (Id - A_skew) / (Id + A_skew)
    return W
end

function (m::CayleyDense)(x::AbstractVecOrMat)
    W = cayley_weight(m)
    return m.σ.(W * x .+ m.bias)
end

function (m::CayleyDense)(x::AbstractArray)
    sz = size(x)
    x2 = reshape(x, sz[1], :)
    y  = m(x2)
    return reshape(y, size(y, 1), sz[2:end]...)
end

function Base.show(io::IO, m::CayleyDense)
    σ_name = m.σ === identity ? "" : ", $(m.σ)"
    print(io, "CayleyDense($(m.n) => $(m.n)$(σ_name))")
end

#  HouseholderLayer – orthogonal via product of Householder reflections
"""
    HouseholderLayer(in => out, σ = identity;
                     n_reflections = out, bias = true, T = Float32)

A *fully* orthogonal layer parametrised as a product of `n_reflections`
Householder reflections.

Each reflection is defined by a learnable vector `vₖ`:

    Hₖ = I − 2 vₖ vₖᵀ / (vₖᵀ vₖ)

The weight matrix is `W = (H₁ H₂ ⋯ Hₖ)[:, 1:out]` when `in ≥ out`,
which is always (semi-)orthogonal.

Supports non-square transformations (`in ≥ out` required).

# Arguments
- `in::Int`:  Input  dimension.
- `out::Int`: Output dimension (`out ≤ in`).
- `σ`:        Activation function (default `identity`).

# Keyword Arguments
- `n_reflections::Int`: Number of Householder reflections
  (default `out`, maximum `in`).
- `bias::Bool`: Include a bias (default `true`).
- `T::Type`:    Element type (default `Float32`).

# Example
```julia
layer = HouseholderLayer(256 => 128, relu)
y = layer(randn(Float32, 256, 32))
```
"""
struct HouseholderLayer{F, M <: AbstractMatrix, B}
    V::M        # each column is a Householder vector (in × n_reflections)
    bias::B
    σ::F
    in_dim::Int
    out_dim::Int
end

Flux.@layer HouseholderLayer

function HouseholderLayer((in_dim, out_dim)::Pair{Int,Int}, σ = identity;
                          n_reflections::Int = -1,
                          bias::Bool = true,
                          T::Type{<:AbstractFloat} = Float32)
    n_reflections = n_reflections < 0 ? out_dim : n_reflections
    in_dim ≥ out_dim || throw(DimensionMismatch(
        "HouseholderLayer requires in ≥ out (got $in_dim => $out_dim). " *
        "Transpose your problem or use OrthogonalDense."))
    k = min(n_reflections, in_dim)
    # Initialise Householder vectors randomly
    V = randn(T, in_dim, k)
    b = bias ? zeros(T, out_dim) : Flux.Zeros()
    return HouseholderLayer(V, b, σ, in_dim, out_dim)
end

"""
    householder_weight(m::HouseholderLayer)

Materialise the (semi-)orthogonal weight matrix from the product of
Householder reflections defined by the columns of `m.V`.

Returns a matrix of size `out × in`.
"""
function householder_weight(m::HouseholderLayer)
    T  = eltype(m.V)
    n  = m.in_dim
    k  = size(m.V, 2)
    W  = Matrix{T}(I, n, n)
    for i in 1:k
        v   = m.V[:, i]
        nrm = dot(v, v)
        if nrm > eps(T)
            W = W - (2 / nrm) * (W * v) * v'
        end
    end
    # Slice to out_dim rows (transpose to get out × in)
    return W[1:m.out_dim, :]
end

function (m::HouseholderLayer)(x::AbstractVecOrMat)
    W = householder_weight(m)
    return m.σ.(W * x .+ m.bias)
end

function (m::HouseholderLayer)(x::AbstractArray)
    sz = size(x)
    x2 = reshape(x, sz[1], :)
    y  = m(x2)
    return reshape(y, size(y, 1), sz[2:end]...)
end

function Base.show(io::IO, m::HouseholderLayer)
    k = size(m.V, 2)
    σ_name = m.σ === identity ? "" : ", $(m.σ)"
    print(io, "HouseholderLayer($(m.in_dim) => $(m.out_dim)$(σ_name); reflections=$k)")
end
