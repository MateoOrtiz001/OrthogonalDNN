using LinearAlgebra
using Random

function house(x)
    n = size(x,1) #(m,n)
    s = x[2:n]'x[2:n]
    v = [1; x[2:n]]

    if s == 0
        β = 0
    else
        mu = sqrt(x[1]^2 + s)
        if x[1] <= 0
            v[1] = x[1] - mu
        else
            v[1] = -s /(x[1]+mu)
        end
        β = 2((v[1])^2)/(s + (v[1])^2)
        v=v/v[1]
    end
    return v, β
end

function VβHouse(A)
    m,n = size(A)
    β   = zeros(n,1)
    Y   = zeros(m,n)

    Am  = copy(A)

    # Primera actualización
    v1,β[1] = house(Am[:,1])
    Y[:,1]  = v1
    W       = -β[1]*v1
    Am      = ( UniformScaling(1) - β[1]*v1*v1')Am

    #De ahi para adelante
    for j = 2:n
        v1,β[j] = house(Am[j:m,j])
        Y[:,j]  = [ zeros(j-1,1) ; v1 ]
        z       = -β[j]*(UniformScaling(1)+W*Y[:,1:j-1]')*Y[:,j]
        W       = [ W z ]
        Am[j:m,j:n] = ( UniformScaling(1) - β[j]*v1*v1')Am[j:m,j:n]
    end
    return Y,W,Am
end

function orthogonal_matrix(rows::Int, cols::Int, seed::Union{Nothing,Integer}=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    if rows ≥ cols               
        A = randn(rng,Float32, rows, cols) * √2f0
        Y, W, _ = VβHouse(A)
        Q = UniformScaling(1) + W*Y'
        return Q[:, 1:cols]       
    else                          
        A = randn(rng,Float32, cols, rows) * √2f0     
        Y, W, _ = VβHouse(A)
        Q = UniformScaling(1) + W*Y'
        Q = Q[:, 1:rows]         
        return Q'                
    end
end