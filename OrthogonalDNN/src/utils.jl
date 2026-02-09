# ──────────────────────────────────────────────────────────────────────────────
# utils.jl – Evaluation utilities
# ──────────────────────────────────────────────────────────────────────────────

"""
    accuracy(data_loader, model)

Compute classification accuracy over batches yielded by `data_loader`.
Assumes one-hot encoded targets.

# Arguments
- `data_loader`: A `Flux.DataLoader` yielding `(x, y)` pairs.
- `model`: A Flux model.

# Returns
- `acc::Float64`: Fraction of correctly classified samples.
"""
function accuracy(data_loader, model)
    correct = 0
    total   = 0
    for (x, y) in data_loader
        ŷ       = model(x)
        correct += sum(Flux.onecold(ŷ) .== Flux.onecold(y))
        total   += size(y, 2)
    end
    return correct / total
end

"""
    orthogonality_error(W::AbstractMatrix)

Measure how far `W` is from having orthonormal columns:

    ‖WᵀW − I‖₂   (spectral norm)

A value close to zero indicates near-perfect orthogonality.
"""
function orthogonality_error(W::AbstractMatrix)
    n = size(W, 2)
    return opnorm(W' * W - I(n), 2)
end

"""
    layer_orthogonality_errors(model)

Return a vector of orthogonality errors for each `Dense` or
`OrthogonalDense` layer in `model`.
"""
function layer_orthogonality_errors(model)
    errors = Float64[]
    for layer in Flux.modules(model)
        if layer isa Dense || layer isa OrthogonalDense
            push!(errors, orthogonality_error(layer.weight))
        end
    end
    return errors
end

"""
    layer_condition_numbers(model)

Return a vector of condition numbers κ(W) = σ_max / σ_min for each
`Dense` or `OrthogonalDense` layer in `model`.
"""
function layer_condition_numbers(model)
    conds = Float64[]
    for layer in Flux.modules(model)
        if layer isa Dense || layer isa OrthogonalDense
            push!(conds, cond(layer.weight))
        end
    end
    return conds
end
