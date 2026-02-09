# OrthogonalDNN.jl

A Julia package for building deep neural networks with **orthogonal weight matrices** using [Flux.jl](https://fluxml.ai/).

Orthogonal weights preserve distances and angles between representations across layers, which helps:

- **Mitigate vanishing/exploding gradients** during back-propagation.
- **Improve generalisation** by acting as a geometric regulariser.
- **Maintain numerical stability** (condition number κ(W) ≈ 1).

## Installation

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/MateoOrtiz001/OrthogonalDNN", subdir="OrthogonalDNN"))
```

Or in development mode:

```julia
using Pkg
Pkg.develop(PackageSpec(path="path/to/OrthogonalDNN", subdir="OrthogonalDNN"))
```

## Quick Start

```julia
using Flux, OrthogonalDNN

# Build a model with orthogonal layers
model = Chain(
    Flux.flatten,
    OrthogonalDense(784 => 256, relu),
    OrthogonalDense(256 => 128, relu),
    CayleyDense(128 => 128, relu),        # exactly orthogonal (square)
    OrthogonalDense(128 => 10),
    softmax
)

# Training with orthogonal regularisation
λ = 1e-4
loss(x, y) = total_loss(model(x), y, model; λ = λ)
```

## Layer Types

| Layer | Orthogonality | Square only? | Description |
|---|---|---|---|
| `OrthogonalDense` | At init (soft) | No | Dense layer initialised with orthogonal weights. Pair with a regulariser. |
| `CayleyDense` | Exact (hard) | **Yes** | Cayley transform of a skew-symmetric matrix: W = (I−A)(I+A)⁻¹. |
| `HouseholderLayer` | Exact (hard) | No (`in ≥ out`) | Product of learnable Householder reflections. |

### OrthogonalDense

```julia
layer = OrthogonalDense(784 => 256, relu)           # bias = true by default
layer = OrthogonalDense(784 => 256, relu; bias=false)
```

### CayleyDense

```julia
layer = CayleyDense(128 => 128, relu)
W = cayley_weight(layer)   # materialise the orthogonal weight matrix
```

### HouseholderLayer

```julia
layer = HouseholderLayer(256 => 128, relu; n_reflections=128)
W = householder_weight(layer)
```

## Regularisers

```julia
# Frobenius penalty:  ‖WᵀW − I‖²_F
orthogonal_regularizer(W)

# Spectral penalty:  (σ_max(W) − 1)²
spectral_regularizer(W)

# Mutual coherence:  max_{i≠j} |⟨wᵢ,wⱼ⟩|/(‖wᵢ‖‖wⱼ‖)
mutual_coherence_regularizer(W)
```

## Utilities

```julia
# Flux-compatible orthogonal initialiser
Dense(128 => 64; init = orthogonal_init)

# Numerical diagnostics
layer_orthogonality_errors(model)   # ‖WᵀW − I‖₂ per layer
layer_condition_numbers(model)      # κ(W) per layer

# Evaluation
accuracy(test_loader, model)
```

## Householder Routines

The package exposes the underlying numerical routines:

```julia
v, β = house(x)               # single Householder reflection
Q, R = qr_householder(A)      # explicit-Q QR factorisation
Y, W, R = wy_householder(A)   # compact-WY representation
Q = orthogonal_matrix(m, n)   # generate (semi-)orthogonal matrix
```

## References

1. S. Li, K. Jia, Y. Wen, T. Liu, D. Tao, "Orthogonal Deep Neural Networks", *IEEE TPAMI*, 2021.
2. S. J. D. Prince, *Understanding Deep Learning*, MIT Press, 2023.

## License

MIT
