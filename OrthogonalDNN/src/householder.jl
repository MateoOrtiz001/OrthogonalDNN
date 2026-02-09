# ──────────────────────────────────────────────────────────────────────────────
# householder.jl – Householder reflection algorithms
#
# Provides the core numerical routines for computing Householder reflections,
# QR factorisations (both the explicit-Q form and the compact-WY
# representation), and related utilities.
# ──────────────────────────────────────────────────────────────────────────────

"""
    house(x::AbstractVector)

Compute the Householder vector `v` and scalar `β` such that

    (I - β v vᵀ) x = ‖x‖ e₁

The returned vector `v` is normalised so that `v[1] = 1`.

# Arguments
- `x`: Input vector of length `n`.

# Returns
- `v::Vector`: Householder vector (length `n`, `v[1] = 1`).
- `β::Real`: Householder scalar.

# References
Golub & Van Loan, *Matrix Computations*, 4th ed., Algorithm 5.1.1.
"""
function house(x::AbstractVector)
    n = length(x)
    s = dot(x[2:n], x[2:n])
    v = [one(eltype(x)); x[2:n]]

    if s ≈ 0
        β = zero(eltype(x))
    else
        μ = sqrt(x[1]^2 + s)
        if x[1] ≤ 0
            v[1] = x[1] - μ
        else
            v[1] = -s / (x[1] + μ)
        end
        β = 2 * v[1]^2 / (s + v[1]^2)
        v = v / v[1]
    end
    return v, β
end

"""
    qr_householder(A::AbstractMatrix)

Compute the explicit QR factorisation of `A` via successive Householder
reflections.

Returns `(Q, R)` where `Q` is orthogonal (or unitary) and `R` is upper
triangular.

# Arguments
- `A`: An `m × n` matrix with `m ≥ n`.

# Returns
- `Q::Matrix`: Orthogonal matrix of size `m × m`.
- `R::Matrix`: Upper triangular matrix of size `m × n`.
"""
function qr_householder(A::AbstractMatrix)
    m, n = size(A)
    Q = Matrix{eltype(A)}(I, m, m)
    R = copy(convert(Matrix{eltype(A)}, A))

    for j in 1:min(m, n)
        v, β = house(R[j:m, j])
        # Pad v to full size
        v_full = zeros(eltype(A), m)
        v_full[j:m] .= v
        Qj = I - β * v_full * v_full'
        Q = Q * Qj
        R = Qj * R
    end
    return Q, R
end

"""
    wy_householder(A::AbstractMatrix)

Compact WY representation of the Householder QR factorisation.

Computes matrices `Y`, `W` such that `Q = I + W Yᵀ`, together with
the upper-triangular factor `R`.  This representation is numerically
more stable and efficient for accumulating many Householder reflections.

# Arguments
- `A`: An `m × n` matrix with `m ≥ n`.

# Returns
- `Y::Matrix`: Matrix of Householder vectors (`m × n`).
- `W::Matrix`: WY multiplier (`m × n`).
- `R::Matrix`: Upper triangular factor (`m × n`).

# References
Schreiber & Van Loan, "A storage-efficient WY representation for products
of Householder transformations", *SIAM J. Sci. Stat. Comput.*, 1989.
"""
function wy_householder(A::AbstractMatrix)
    m, n = size(A)
    T     = eltype(A)
    β     = zeros(T, n)
    Y     = zeros(T, m, n)
    Am    = copy(convert(Matrix{T}, A))

    # First column
    v1, β[1] = house(Am[:, 1])
    Y[:, 1]  = v1
    W        = -β[1] * v1
    Am       = (I - β[1] * v1 * v1') * Am

    # Remaining columns
    for j in 2:n
        v, β[j] = house(Am[j:m, j])
        Y[:, j] = [zeros(T, j - 1); v]
        z       = -β[j] * (I + W * Y[:, 1:j-1]') * Y[:, j]
        W       = [W z]
        Am[j:m, j:n] = (I - β[j] * v * v') * Am[j:m, j:n]
    end
    return Y, W, Am
end
