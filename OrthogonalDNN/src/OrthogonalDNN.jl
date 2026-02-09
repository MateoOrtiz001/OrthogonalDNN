"""
    OrthogonalDNN

A Julia package for building deep neural networks with orthogonal weight
matrices using [Flux.jl](https://fluxml.ai/).

Provides three types of orthogonal layers:

- **`OrthogonalDense`** – Dense layer *initialised* with orthogonal weights.
  Pair with a regulariser to maintain orthogonality during training.
- **`CayleyDense`** – Fully orthogonal square layer via the Cayley transform.
  Weights are exactly orthogonal by construction at every training step.
- **`HouseholderLayer`** – Fully orthogonal layer parametrised as a product
  of Householder reflections. Supports non-square (`in ≥ out`)
  transformations.

Also includes orthogonal regularisers, a Flux-compatible weight initialiser,
and numerical analysis utilities.
"""
module OrthogonalDNN

using Flux
using LinearAlgebra
using Random

# Internal numeric routines
include("householder.jl")

# Weight initialisation 
include("orthogonal_init.jl")

# Flux layers 
include("layers.jl")

# Regularisation 
include("regularization.jl")

# Utilities (accuracy, diagnostics) 
include("utils.jl")

####
# Householder routines
export house, qr_householder, wy_householder

# Initialiser
export orthogonal_matrix, orthogonal_init

# Layers
export OrthogonalDense, CayleyDense, HouseholderLayer
export cayley_weight, householder_weight

# Regularisers
export orthogonal_regularizer, mutual_coherence_regularizer, spectral_regularizer
export get_weight_matrices, total_loss

# Utilities
export accuracy, orthogonality_error
export layer_orthogonality_errors, layer_condition_numbers

end # module OrthogonalDNN
