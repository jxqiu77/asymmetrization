using Arpack
using DelimitedFiles
using LinearAlgebra
using Random

const ASYMMETRIZATION_EIGS_TOL = 1.0e-8
const ASYMMETRIZATION_EIGS_MAXITER = 5000
const ASYMMETRIZATION_DENSE_CUTOFF = 64
const ASYMMETRIZATION_REAL_TOL = 1.0e-7
const ASYMMETRIZATION_MATCH_TOL = 1.0e-5

"""A selected finite-sample outlier lies genuinely off the real axis."""
struct NonrealOutlierError <: Exception
    eigenvalue::ComplexF64
end

function Base.showerror(io::IO, error_value::NonrealOutlierError)
    eigenvalue = error_value.eigenvalue
    tolerance = ASYMMETRIZATION_REAL_TOL * max(1.0, abs(eigenvalue))
    print(
        io,
        "selected outlier eigenvalue $(eigenvalue) has |Im(lambda)|=$(abs(imag(eigenvalue))), " *
        "which exceeds the realness tolerance $(tolerance)",
    )
end

"""
    make_block_variance_profile(row_count, column_count, kappa)

Construct the block profile used in the numerical simulations. The Wigner
case is obtained by setting `row_count == column_count`.
"""
function make_block_variance_profile(
    row_count::Int,
    column_count::Int,
    kappa::Real,
)
    iseven(row_count) || error("the block variance profile requires an even row count")
    iseven(column_count) || error("the block variance profile requires an even column count")
    kappa >= 1.0 || error("kappa must be at least one")

    profile = fill(1.0 / column_count, row_count, column_count)
    profile[1:div(row_count, 2), 1:div(column_count, 2)] .=
        kappa / column_count
    return profile
end

make_rectangular_variance_profile(p::Int, n::Int, kappa::Real) =
    make_block_variance_profile(p, n, kappa)

"""
    variance_profile_opnorm(T)

Return `||T||_2`, which determines the predicted bulk edge.
"""
variance_profile_opnorm(T::AbstractMatrix{<:Real}) = opnorm(T, 2)

"""
    make_wigner(n; variance_profile, law=:gaussian, rng=Random.default_rng())

Generate a symmetric Wigner-type matrix with the requested variance profile and
unit-variance entry law.
"""
function make_wigner(
    n::Int;
    variance_profile,
    law::Symbol=:gaussian,
    rng::AbstractRNG=Random.default_rng(),
)
    W = zeros(Float64, n, n)

    @inbounds for j in 1:n
        for i in 1:j
            value = sqrt(variance_profile[i, j]) * draw_unit_variance_noise(rng, law)
            W[i, j] = value
            W[j, i] = value
        end
    end

    return W
end

"""
    draw_unit_variance_noise(rng, law)

Draw one scalar from the requested centered, unit-variance noise law.
"""
function draw_unit_variance_noise(rng::AbstractRNG, law::Symbol)
    if law == :gaussian
        return randn(rng)
    elseif law == :rademacher
        return rand(rng, Bool) ? 1.0 : -1.0
    else
        error("unknown noise law: $(law)")
    end
end

"""
    cosine_spike_vectors(n, r)

Construct the delocalized cosine spike vectors from the paper.
"""
function cosine_spike_vectors(n::Int, r::Int)
    U = zeros(Float64, n, r)
    r == 0 && return U

    U[:, 1] .= inv(sqrt(n))

    @inbounds for k in 1:(r - 1)
        scale = sqrt(2 / n)
        for i in 1:n
            U[i, k + 1] = scale * cos(pi * k * (i - 0.5) / n)
        end
    end

    return U
end

"""
    block_spike_vectors(dimension, count)

Construct the block-oriented cosine directions used in the numerical
simulations, with squared masses `0.9` and `0.1` in the two halves.
"""
function block_spike_vectors(dimension::Int, count::Int)
    iseven(dimension) || error("the block spike construction requires an even dimension")
    split = div(dimension, 2)
    1 <= count <= split || error("count must lie between one and half the dimension")
    basis = cosine_spike_vectors(split, count)
    return vcat(sqrt(0.9) .* basis, sqrt(0.1) .* basis)
end

rectangular_spike_vectors(dimension::Int, count::Int) =
    block_spike_vectors(dimension, count)

"""
    make_signal_matrix(U, d)

Construct `S = sum_k d_k u_k u_k'`.
"""
function make_signal_matrix(U::AbstractMatrix{<:Real}, d::AbstractVector{<:Real})
    size(U, 2) == length(d) || error("dimension mismatch between spike vectors and strengths")
    return Matrix(Symmetric(U * Diagonal(collect(Float64, d)) * U'))
end


"""
    make_signal_matrix(U, d, V)

Construct the rectangular signal `U * Diagonal(d) * V'`.
"""
function make_signal_matrix(
    U::AbstractMatrix{<:Real},
    d::AbstractVector{<:Real},
    V::AbstractMatrix{<:Real},
)
    size(U, 2) == length(d) ||
        error("dimension mismatch between left vectors and strengths")
    size(V, 2) == length(d) ||
        error("dimension mismatch between right vectors and strengths")
    return Matrix(U * Diagonal(collect(Float64, d)) * transpose(V))
end

make_rectangular_signal_matrix(U, d, V) = make_signal_matrix(U, d, V)

"""
    make_rectangular_noise(p, n; variance_profile, law=:gaussian, rng)

Generate a rectangular matrix with independent centered entries and the
requested variance profile.
"""
function make_rectangular_noise(
    p::Int,
    n::Int;
    variance_profile,
    law::Symbol=:gaussian,
    rng::AbstractRNG=Random.default_rng(),
)
    size(variance_profile) == (p, n) ||
        error("variance_profile must have size ($(p), $(n))")

    X = Matrix{Float64}(undef, p, n)
    @inbounds for j in 1:n, i in 1:p
        X[i, j] = sqrt(variance_profile[i, j]) *
                  draw_unit_variance_noise(rng, law)
    end
    return X
end

"""
    make_symmetric_mask(n; p=0.5, rng=Random.default_rng())

Generate the symmetric Bernoulli mask from the paper.
"""
function make_symmetric_mask(n::Int; p::Float64=0.5, rng::AbstractRNG=Random.default_rng())
    P = zeros(Float64, n, n)

    @inbounds for j in 1:n
        for i in 1:j
            value = rand(rng) <= p ? 1.0 : 0.0
            P[i, j] = value
            P[j, i] = value
        end
    end

    return P
end

"""
    make_rectangular_mask(p, n; rng=Random.default_rng())

Draw the iid Bernoulli(1/2) mask used by Algorithms 3 and 4.
"""
function make_rectangular_mask(
    p::Int,
    n::Int;
    rng::AbstractRNG=Random.default_rng(),
)
    p > 0 || error("p must be positive")
    n > 0 || error("n must be positive")
    return BitMatrix(rand(rng, p, n) .<= 0.5)
end

"""
    make_block_matrix(Y, P)

Construct

    [0                       P .* Y
     ((1 - P) .* Y)'        0      ].

For symmetric `Y` and `P`, this is the Wigner split matrix because the
lower-left block is itself symmetric.
"""
function make_block_matrix(Y::AbstractMatrix{<:Real}, P::AbstractMatrix{<:Real})
    size(Y) == size(P) || error("Y and P must have the same size")
    p, n = size(Y)
    split_matrix = zeros(Float64, p + n, p + n)
    split_matrix[1:p, (p + 1):(p + n)] .= P .* Y
    split_matrix[(p + 1):(p + n), 1:p] .= transpose((1 .- P) .* Y)
    return split_matrix
end

function _eigenvalue_order(values, which::Symbol)
    if which == :LR
        return sortperm(real.(values); rev=true)
    elseif which == :SR
        return sortperm(real.(values))
    elseif which == :LM
        return sortperm(abs.(values); rev=true)
    end
    error("unsupported eigenvalue selector: $(which)")
end

function _dense_extreme_eigenpairs(A, nev, which; ritzvec)
    decomposition = eigen(Matrix{Float64}(A))
    indices = _eigenvalue_order(decomposition.values, which)[1:nev]
    vectors = ritzvec ? decomposition.vectors[:, indices] : nothing
    return decomposition.values[indices], vectors
end

"""
    extreme_eigenpairs(A, nev, which, rng; solver=:auto, ritzvec=true)

Compute extreme eigenpairs with ARPACK, using a dense solve for small matrices
and as a fallback after ARPACK failure or non-convergence.
"""
function extreme_eigenpairs(
    A::AbstractMatrix{<:Real},
    nev::Int,
    which::Symbol,
    rng::AbstractRNG;
    solver::Symbol=:auto,
    ritzvec::Bool=true,
)
    size(A, 1) == size(A, 2) || error("A must be square")
    dimension = size(A, 1)
    1 <= nev <= dimension || error("nev must be between 1 and $(dimension)")
    solver in (:auto, :arpack, :dense) ||
        error("solver must be :auto, :arpack, or :dense")

    use_dense = solver == :dense ||
                (solver == :auto &&
                 (dimension <= ASYMMETRIZATION_DENSE_CUTOFF || nev >= dimension - 1))
    if use_dense
        values, vectors = _dense_extreme_eigenpairs(A, nev, which; ritzvec=ritzvec)
        return (
            values=values,
            vectors=vectors,
            used_fallback=false,
            fallback_reason=nothing,
        )
    end

    fallback_reason = nothing
    try
        result = eigs(
            A;
            nev=nev,
            which=which,
            tol=ASYMMETRIZATION_EIGS_TOL,
            maxiter=ASYMMETRIZATION_EIGS_MAXITER,
            v0=randn(rng, dimension),
            ritzvec=ritzvec,
        )
        values = result[1]
        vectors = ritzvec ? result[2] : nothing
        nconv = ritzvec ? result[3] : result[2]
        if nconv >= nev
            indices = _eigenvalue_order(values, which)[1:nev]
            return (
                values=values[indices],
                vectors=ritzvec ? vectors[:, indices] : nothing,
                used_fallback=false,
                fallback_reason=nothing,
            )
        end
        fallback_reason = "ARPACK converged $(nconv) of $(nev) eigenvalues"
    catch error_value
        error_value isa InterruptException && rethrow()
        fallback_reason = sprint(showerror, error_value)
    end

    values, vectors = _dense_extreme_eigenpairs(A, nev, which; ritzvec=ritzvec)
    return (
        values=values,
        vectors=vectors,
        used_fallback=true,
        fallback_reason=fallback_reason,
    )
end

function _match_eigenvalues(target_values, candidate_values)
    length(target_values) == length(candidate_values) ||
        error("target and candidate eigenvalue counts must agree")

    matched_indices = zeros(Int, length(target_values))
    available = trues(length(candidate_values))
    for i in eachindex(target_values)
        best_index = 0
        best_distance = Inf
        for j in eachindex(candidate_values)
            available[j] || continue
            distance = abs(candidate_values[j] - target_values[i])
            if distance < best_distance
                best_distance = distance
                best_index = j
            end
        end
        best_index > 0 || error("failed to match left and right eigenvalues")
        scale = max(1.0, abs(target_values[i]), abs(candidate_values[best_index]))
        best_distance <= ASYMMETRIZATION_MATCH_TOL * scale ||
            error("left and right eigenvalue computations do not match")
        matched_indices[i] = best_index
        available[best_index] = false
    end
    return matched_indices
end

function _real_eigenvector(vector, eigenvalue)
    if abs(imag(eigenvalue)) >
       ASYMMETRIZATION_REAL_TOL * max(1.0, abs(eigenvalue))
        throw(NonrealOutlierError(ComplexF64(eigenvalue)))
    end

    pivot = argmax(abs.(vector))
    pivot_magnitude = abs(vector[pivot])
    pivot_magnitude > ASYMMETRIZATION_REAL_TOL ||
        error("a selected outlier eigenvector is numerically zero")
    rotated = vector .* conj(vector[pivot] / pivot_magnitude)
    norm(imag.(rotated)) <=
        ASYMMETRIZATION_REAL_TOL * max(1.0, norm(rotated)) ||
        error("a selected outlier eigenvector is not real")
    return real.(rotated)
end

"""
    split_outlier_factors(split_matrix, first_block_size, r, rng; solver=:auto)

Compute and normalize the positive-outlier left/right eigenvectors of a split
matrix. By the exact identity `J * split_matrix * J = -split_matrix`, the
sum of the positive and negative projector blocks equals twice the outer
product of the corresponding positive-outlier blocks.
"""
function split_outlier_factors(
    split_matrix::AbstractMatrix{<:Real},
    first_block_size::Int,
    r::Int,
    rng::AbstractRNG;
    solver::Symbol=:auto,
)
    dimension = size(split_matrix, 1)
    size(split_matrix, 2) == dimension || error("split_matrix must be square")
    1 <= first_block_size < dimension || error("invalid first block size")

    right_result = extreme_eigenpairs(
        split_matrix,
        r,
        :LR,
        rng;
        solver=solver,
        ritzvec=true,
    )
    left_result = extreme_eigenpairs(
        transpose(split_matrix),
        r,
        :LR,
        rng;
        solver=solver,
        ritzvec=true,
    )

    right_order = sortperm(real.(right_result.values); rev=true)
    values = right_result.values[right_order]
    right_vectors = right_result.vectors[:, right_order]
    left_indices = _match_eigenvalues(values, left_result.values)
    left_vectors = left_result.vectors[:, left_indices]

    second_block_size = dimension - first_block_size
    right_first = zeros(Float64, first_block_size, r)
    right_second = zeros(Float64, second_block_size, r)
    left_first = zeros(Float64, first_block_size, r)
    left_second = zeros(Float64, second_block_size, r)

    for k in 1:r
        right = _real_eigenvector(right_vectors[:, k], values[k])
        left = _real_eigenvector(left_vectors[:, k], values[k])
        right ./= norm(right)
        inner_product = dot(left, right)
        abs(inner_product) > ASYMMETRIZATION_REAL_TOL ||
            error("left and right outlier eigenvectors are nearly orthogonal")
        left ./= inner_product

        first_nonzero = findfirst(value -> abs(value) > ASYMMETRIZATION_REAL_TOL, right)
        first_nonzero === nothing && error("a selected outlier eigenvector is zero")
        if right[first_nonzero] < 0.0
            right .*= -1.0
            left .*= -1.0
        end

        right_first[:, k] .= right[1:first_block_size]
        right_second[:, k] .= right[(first_block_size + 1):dimension]
        left_first[:, k] .= left[1:first_block_size]
        left_second[:, k] .= left[(first_block_size + 1):dimension]
    end

    return (
        estimates=2.0 .* real.(values),
        positive_eigenvalues=values,
        negative_eigenvalues=-values,
        right_first=right_first,
        right_second=right_second,
        left_first=left_first,
        left_second=left_second,
        used_fallback=right_result.used_fallback || left_result.used_fallback,
        fallback_reasons=(right_result.fallback_reason, left_result.fallback_reason),
    )
end

function validate_rectangular_algorithm_input(Y, r)
    p, n = size(Y)
    p > 0 || error("Y must have at least one row")
    n > 0 || error("Y must have at least one column")
    1 <= r <= min(p, n) || error("r must be between 1 and min(p, n)")
    all(isfinite, Y) || error("Y must contain only finite values")
    return p, n
end

function validate_wigner_algorithm_input(Y, r)
    n, m = size(Y)
    n == m || error("Y must be square")
    issymmetric(Y) || error("Y must be symmetric")
    1 <= r <= n || error("r must be between 1 and n")
    all(isfinite, Y) || error("Y must contain only finite values")
    return n
end

function rectangular_row_count(n::Int)
    n > 0 || error("n must be positive")
    mod(3 * n, 5) == 0 || error("the simulation design requires 3n/5 to be an integer")
    return div(3 * n, 5)
end

positive_finite_or_zero(value) =
    isfinite(value) && value > 0.0 ? Float64(value) : 0.0

function write_csv_rows(path, header, rows)
    table = Matrix{Any}(undef, length(rows) + 1, length(header))
    table[1, :] .= header
    for (row_index, row) in enumerate(rows)
        length(row) == length(header) || error("CSV row has the wrong length")
        table[row_index + 1, :] .= row
    end
    writedlm(path, table, ',')
end

function center_text(text, width)
    padding = max(width - length(text), 0)
    left_padding = div(padding, 2)
    return repeat(" ", left_padding) * text * repeat(" ", padding - left_padding)
end

"""
    block_eigenvalues(X, P)

Return the eigenvalues of the asymmetrized block matrix as paired square roots
of the eigenvalues of `(P .* X) * ((1 - P) .* X)`.
"""
function block_eigenvalues(X, P)
    A = P .* X
    B = (1.0 .- P) .* X
    roots = sqrt.(complex.(eigvals!(A * B)))
    return vcat(roots, -roots)
end

"""
    padded_limits(values)

Return the extrema of `values` with five percent padding on each side.
"""
function padded_limits(values)
    lower, upper = extrema(values)
    padding = 0.05 * (upper - lower)
    return lower - padding, upper + padding
end
