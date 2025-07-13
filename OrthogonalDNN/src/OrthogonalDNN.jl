module OrthogonalDNN
using Flux
using LinearAlgebra

include("methods.jl")
include("orthogonalLayers.jl")

greet() = print("Hello World!")


end # module OrthogonalDNN
