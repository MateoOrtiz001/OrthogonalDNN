using Test
using LinearAlgebra
using Random
using Flux: relu
using OrthogonalDNN

@testset "OrthogonalDNN.jl" begin

    #  Householder routines
    @testset "house()" begin
        x = [3.0, 1.0, 5.0, 1.0]
        v, β = house(x)
        @test v[1] ≈ 1.0       # normalised so v[1] = 1
        H = I - β * v * v'
        Hx = H * x
        @test abs(Hx[1]) ≈ norm(x) atol = 1e-10
        @test all(abs.(Hx[2:end]) .< 1e-10)
    end

    @testset "qr_householder()" begin
        Random.seed!(42)
        A = randn(6, 4)
        Q, R = qr_householder(A)
        # Q is orthogonal
        @test Q' * Q ≈ I(6) atol = 1e-10
        # R is upper triangular below working block
        for j in 1:4, i in j+1:6
            @test abs(R[i, j]) < 1e-10
        end
        # Reconstruction
        @test Q * R ≈ A atol = 1e-10
    end

    @testset "wy_householder()" begin
        Random.seed!(7)
        A = randn(5, 3)
        Y, W, R = wy_householder(A)
        Q = I + W * Y'
        @test Q' * Q ≈ I(5) atol = 1e-10
        @test Q * R ≈ A atol = 1e-10
    end

    #  Orthogonal matrix generation
    @testset "orthogonal_matrix()" begin
        # Tall (rows ≥ cols): columns should be orthonormal
        Q1 = orthogonal_matrix(8, 5; T = Float64)
        @test size(Q1) == (8, 5)
        @test Q1' * Q1 ≈ I(5) atol = 1e-10

        # Wide (rows < cols): rows should be orthonormal
        Q2 = orthogonal_matrix(3, 7; T = Float64)
        @test size(Q2) == (3, 7)
        @test Q2 * Q2' ≈ I(3) atol = 1e-10

        # Square
        Q3 = orthogonal_matrix(4, 4; T = Float64)
        @test size(Q3) == (4, 4)
        @test Q3' * Q3 ≈ I(4) atol = 1e-10
    end

    @testset "orthogonal_init()" begin
        W = orthogonal_init(5, 3)
        @test eltype(W) == Float32
        @test size(W) == (5, 3)
        # columns approximately orthonormal
        @test W' * W ≈ I(3) atol = 1e-5
    end

    #  Layers
    @testset "OrthogonalDense" begin
        layer = OrthogonalDense(10 => 5, relu)
        x = randn(Float32, 10, 4)
        y = layer(x)
        @test size(y) == (5, 4)
        @test all(y .≥ 0)   # relu
        # Weight is approximately orthogonal at init (5×10: rows are orthonormal)
        @test layer.weight * layer.weight' ≈ I(5) atol = 0.1
    end

    @testset "CayleyDense" begin
        layer = CayleyDense(6 => 6, identity)
        x = randn(Float32, 6, 3)
        y = layer(x)
        @test size(y) == (6, 3)
        # Weight must be orthogonal by construction
        W = cayley_weight(layer)
        @test W' * W ≈ I(6) atol = 1e-4
        # Must fail for non-square
        @test_throws DimensionMismatch CayleyDense(5 => 3)
    end

    @testset "HouseholderLayer" begin
        layer = HouseholderLayer(8 => 5, identity)
        x = randn(Float32, 8, 4)
        y = layer(x)
        @test size(y) == (5, 4)
        # Weight must be semi-orthogonal
        W = householder_weight(layer)
        @test W * W' ≈ I(5) atol = 1e-4
        # Must fail for out > in
        @test_throws DimensionMismatch HouseholderLayer(3 => 5)
    end

    #  Regularisation
    @testset "orthogonal_regularizer()" begin
        Q = Float32.(Matrix(qr(randn(4, 4)).Q))
        @test orthogonal_regularizer(Q) < 1e-5
        A = randn(Float32, 4, 4)
        @test orthogonal_regularizer(A) > 0.1  # not orthogonal
    end

    @testset "spectral_regularizer()" begin
        Q = Float32.(Matrix(qr(randn(5, 5)).Q))
        @test spectral_regularizer(Q) < 1e-5
    end

    #  Utilities
    @testset "orthogonality_error()" begin
        Q = Float64.(Matrix(qr(randn(5, 5)).Q))
        @test orthogonality_error(Q) < 1e-12
    end

end
