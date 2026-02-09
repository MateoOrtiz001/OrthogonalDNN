# ──────────────────────────────────────────────────────────────────────────────
# regularization.jl – Orthogonal regularisation penalties
#
# Provides loss-function terms that penalise deviation from orthogonality,
# to be added to the main task loss during training.
# ──────────────────────────────────────────────────────────────────────────────

"""
    orthogonal_regularizer(W::AbstractMatrix)

Compute the *soft orthogonality* penalty

    ‖WᵀW − I‖²_F

for a weight matrix `W` (2-D) or a reshaped convolutional kernel (4-D).

A value of zero means `W` has perfectly orthonormal columns.

# Example
```julia
model = Chain(OrthogonalDense(784 => 256, relu), OrthogonalDense(256 => 10))
λ = 1e-4
reg = sum(orthogonal_regularizer, get_weight_matrices(model))
loss = crossentropy(model(x), y) + λ * reg
```
"""
function orthogonal_regularizer(W::AbstractMatrix)
    n = size(W, 2)
    return sum(abs2, W' * W - I(n))
end

function orthogonal_regularizer(W::AbstractArray{T,4}) where T
    # Conv kernel: (width, height, channels_in, channels_out)
    W_mat = reshape(W, :, size(W, 4))
    return orthogonal_regularizer(W_mat)
end

"""
    mutual_coherence_regularizer(W::AbstractMatrix)

Compute the *mutual coherence* penalty, defined as the maximum absolute
off-diagonal entry of the Gram matrix WᵀW:

    μ(W) = max_{i ≠ j} |⟨wᵢ , wⱼ⟩| / (‖wᵢ‖ ‖wⱼ‖)

Minimising this encourages the columns of `W` to be as incoherent
(uncorrelated) as possible.
"""
function mutual_coherence_regularizer(W::AbstractMatrix)
    # Normalise columns
    norms = sqrt.(sum(abs2, W; dims = 1))
    Wn    = W ./ max.(norms, eps(eltype(W)))
    G     = Wn' * Wn
    n     = size(G, 1)
    # Zero out diagonal
    mask  = ones(eltype(G), n, n) - I(n)
    return maximum(abs.(G .* mask))
end

"""
    spectral_regularizer(W::AbstractMatrix; target = 1)

Penalise deviation of the largest singular value from `target`:

    (σ_max(W) − target)²

This is lighter than the full Frobenius penalty and focuses on preventing
gradient explosion (σ_max ≫ 1) or vanishing (σ_max ≪ 1).
"""
function spectral_regularizer(W::AbstractMatrix; target::Real = 1)
    σ_max = opnorm(W, 2)
    return (σ_max - target)^2
end

#  Utility: extract weight matrices from a model

"""
    get_weight_matrices(model)

Return a vector of all weight matrices found in `model` that belong to
orthogonal-aware layers (`Dense`, `OrthogonalDense`).and convolutional layers.
"""
function get_weight_matrices(model)
    weights = AbstractMatrix[]
    for layer in Flux.modules(model)
        if layer isa Dense || layer isa OrthogonalDense
            push!(weights, layer.weight)
        elseif layer isa Conv
            push!(weights, reshape(layer.weight, :, size(layer.weight, 4)))
        end
    end
    return weights
end

"""
    total_loss(ŷ, y, model; λ = 1e-4, loss_fn = Flux.crossentropy,
               regularizer = orthogonal_regularizer)

Convenience function that combines a supervised loss with orthogonal
regularisation over all weight matrices in `model`.

# Arguments
- `ŷ`: Model predictions.
- `y`: Target labels.
- `model`: Flux model.

# Keyword Arguments
- `λ::Real`: Regularisation strength (default `1e-4`).
- `loss_fn`: Task loss function (default `Flux.crossentropy`).
- `regularizer`: Per-matrix penalty (default [`orthogonal_regularizer`](@ref)).

# Returns
Scalar loss value.
"""
function total_loss(ŷ, y, model;
                    λ::Real = 1e-4,
                    loss_fn = Flux.crossentropy,
                    regularizer = orthogonal_regularizer)
    ce = loss_fn(ŷ, y)
    if λ ≈ 0
        return ce
    end
    ws = get_weight_matrices(model)
    reg = isempty(ws) ? zero(ce) : sum(regularizer, ws)
    return ce + eltype(ce)(λ) * reg
end
