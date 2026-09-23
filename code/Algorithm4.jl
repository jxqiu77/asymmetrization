#=
Implementation and numerical simulation for Algorithm 4.

Usage:
  julia --project=. code/Algorithm4.jl
=#

using DelimitedFiles
using LinearAlgebra
using Printf
using ProgressMeter
using Random
using Statistics

isdefined(@__MODULE__, :ASYMMETRIZATION_EIGS_TOL) ||
    include(joinpath(@__DIR__, "asymmetrization_utils.jl"))

const ALGORITHM4_BASE_SEED = 20260813
const ALGORITHM4_MAX_MASK_ATTEMPTS = 10

struct Algorithm4MaskRetryError <: Exception
    sample::Int
    eigenvalue::ComplexF64
    total_mask_retries::Int
    max_mask_attempts::Int
end

function Base.showerror(io::IO, error_value::Algorithm4MaskRetryError)
    print(
        io,
        "Algorithm 4 sample $(error_value.sample) still selected the non-real " *
        "outlier $(error_value.eigenvalue) after " *
        "$(error_value.max_mask_attempts) mask attempts " *
        "($(error_value.total_mask_retries) total retries in this call)",
    )
end

"""
    rectangular_outlier_factors(Y, P, r, eigs_rng; solver=:auto)

Compute the positive-outlier left/right factors for one fixed split. The
negative factors prescribed by Algorithm 4 are obtained exactly as `J*r` and
`J*l`: the split dilation obeys `J*M*J = -M`, so this is the same paired
negative outlier without a redundant eigensolver call.
"""
function rectangular_outlier_factors(
    Y::AbstractMatrix{<:Real},
    P::AbstractMatrix{Bool},
    r::Int,
    eigs_rng::AbstractRNG;
    solver::Symbol=:auto,
)
    p, n = validate_rectangular_algorithm_input(Y, r)
    size(P) == (p, n) || error("Y and P must have the same size")
    factors = split_outlier_factors(
        make_block_matrix(Y, P),
        p,
        r,
        eigs_rng;
        solver=solver,
    )
    return (
        estimates=factors.estimates,
        positive_eigenvalues=factors.positive_eigenvalues,
        negative_eigenvalues=factors.negative_eigenvalues,
        right_U=factors.right_first,
        right_V=factors.right_second,
        left_U=factors.left_first,
        left_V=factors.left_second,
        mask=P,
        used_fallback=factors.used_fallback,
        fallback_reasons=factors.fallback_reasons,
    )
end

function _one_sample_algorithm4(
    Y::AbstractMatrix{<:Real},
    r::Int,
    data_rng::AbstractRNG,
    eigs_rng::AbstractRNG;
    solver::Symbol,
    max_mask_attempts::Int,
    sample_index::Int,
    previous_mask_retries::Int=0,
)
    p, n = validate_rectangular_algorithm_input(Y, r)
    max_mask_attempts >= 1 ||
        throw(ArgumentError("max_mask_attempts must be at least one"))

    for attempt in 1:max_mask_attempts
        P = make_rectangular_mask(p, n; rng=data_rng)
        try
            factors = rectangular_outlier_factors(
                Y,
                P,
                r,
                eigs_rng;
                solver=solver,
            )
            return merge(factors, (mask_retries=attempt - 1,))
        catch error_value
            error_value isa NonrealOutlierError || rethrow()
            if attempt == max_mask_attempts
                throw(Algorithm4MaskRetryError(
                    sample_index,
                    error_value.eigenvalue,
                    previous_mask_retries + attempt - 1,
                    max_mask_attempts,
                ))
            end
        end
    end

    error("unreachable mask-retry state")
end

"""
    rectangular_signal_correlation(d1, d2, R_U, R_V)

Compute the normalized signal correlation from singular strengths and the
left/right overlap matrices.
"""
function rectangular_signal_correlation(d1, d2, R_U, R_V)
    r1 = length(d1)
    r2 = length(d2)
    size(R_U) == (r1, r2) || error("R_U has incompatible dimensions")
    size(R_V) == (r1, r2) || error("R_V has incompatible dimensions")
    denominator = norm(d1) * norm(d2)
    denominator > 0.0 || error("singular-strength vectors must be nonzero")
    weights = d1 * transpose(d2)
    return sum(weights .* R_U .* R_V) / denominator
end

"""
    algorithm4(Y1, Y2, r; rng=Random.default_rng(), eigs_rng=rng,
               solver=:auto, max_mask_attempts=10, return_details=false)

Estimate the left and right overlap matrices, up to their common sign action,
and the normalized signal correlation. For each observation, independently
draw up to `max_mask_attempts` masks until the selected outliers are real. This
mask-retry variant keeps each observation fixed while changing only its
auxiliary splitting mask.
"""
function algorithm4(
    Y1::AbstractMatrix{<:Real},
    Y2::AbstractMatrix{<:Real},
    r::Int;
    rng::AbstractRNG=Random.default_rng(),
    eigs_rng::AbstractRNG=rng,
    solver::Symbol=:auto,
    max_mask_attempts::Int=ALGORITHM4_MAX_MASK_ATTEMPTS,
    return_details::Bool=false,
)
    size(Y1) == size(Y2) || error("Y1 and Y2 must have the same size")
    validate_rectangular_algorithm_input(Y1, r)
    validate_rectangular_algorithm_input(Y2, r)

    sample1 = _one_sample_algorithm4(
        Y1,
        r,
        rng,
        eigs_rng;
        solver=solver,
        max_mask_attempts=max_mask_attempts,
        sample_index=1,
    )
    sample2 = _one_sample_algorithm4(
        Y2,
        r,
        rng,
        eigs_rng;
        solver=solver,
        max_mask_attempts=max_mask_attempts,
        sample_index=2,
        previous_mask_retries=sample1.mask_retries,
    )
    R_U_abs = zeros(Float64, r, r)
    R_V_abs = zeros(Float64, r, r)
    R_U_eqv = zeros(Float64, r, r)
    R_V_eqv = zeros(Float64, r, r)

    # The paired negative eigenvectors are J*r and J*l. Hence the two
    # diagonal projector blocks are 2*xU*yU' and 2*xV*yV', giving these
    # allocation-free trace formulas.
    for i in 1:r, j in 1:r
        cross_U_12 = dot(sample1.left_U[:, i], sample2.right_U[:, j])
        cross_U_21 = dot(sample2.left_U[:, j], sample1.right_U[:, i])
        cross_V_12 = dot(sample1.left_V[:, i], sample2.right_V[:, j])
        cross_V_21 = dot(sample2.left_V[:, j], sample1.right_V[:, i])

        q_U = 4.0 * cross_U_12 * cross_U_21
        q_V = 4.0 * cross_V_12 * cross_V_21
        R_U_abs[i, j] = sqrt(clamp(q_U, 0.0, 1.0))
        R_V_abs[i, j] = sqrt(clamp(q_V, 0.0, 1.0))

        sign_U = cross_U_12 >= 0.0 ? 1.0 : -1.0
        sign_V = cross_V_12 >= 0.0 ? 1.0 : -1.0
        R_U_eqv[i, j] = sign_U * R_U_abs[i, j]
        R_V_eqv[i, j] = sign_V * R_V_abs[i, j]
    end

    rho_sig = rectangular_signal_correlation(
        sample1.estimates,
        sample2.estimates,
        R_U_eqv,
        R_V_eqv,
    )
    result = (R_U_eqv=R_U_eqv, R_V_eqv=R_V_eqv, rho_sig=rho_sig)
    if return_details
        return merge(
            result,
            (
                R_U_abs=R_U_abs,
                R_V_abs=R_V_abs,
                R_UV=R_U_eqv .* R_V_eqv,
                sample1=sample1,
                sample2=sample2,
            ),
        )
    end
    return result
end

function simultaneous_orbit_error(R_U_estimate, R_V_estimate, R_U, R_V)
    size(R_U_estimate) == size(R_U) || error("left overlap dimensions differ")
    size(R_V_estimate) == size(R_V) || error("right overlap dimensions differ")
    size(R_U) == size(R_V) || error("left and right overlap dimensions differ")
    size(R_U, 1) == size(R_U, 2) || error("overlap matrices must be square")

    r = size(R_U, 1)
    error_value = Inf
    for row_state in 0:((1 << r) - 1), column_state in 0:((1 << r) - 1)
        row_signs = [((row_state >> (i - 1)) & 1) == 1 ? 1.0 : -1.0 for i in 1:r]
        column_signs = [
            ((column_state >> (i - 1)) & 1) == 1 ? 1.0 : -1.0 for i in 1:r
        ]
        signed_U = Diagonal(row_signs) * R_U * Diagonal(column_signs)
        signed_V = Diagonal(row_signs) * R_V * Diagonal(column_signs)
        candidate = max(
            maximum(abs.(R_U_estimate .- signed_U)),
            maximum(abs.(R_V_estimate .- signed_V)),
        )
        error_value = min(error_value, candidate)
    end
    return error_value
end

function simulate_algorithm4(
    n_grid,
    B,
    kappa_pairs,
    laws,
    rho_grid,
    data_rng,
    eigs_rng,
)
    rows = Vector{Vector{Any}}()
    r = 2
    A_U = [0.8 0.3; -0.2 0.6]
    A_V = [0.7 -0.1; 0.4 0.6]
    gamma1 = [2.0, 1.6]
    gamma2 = [2.2, 1.7]

    for (kappa1, kappa2) in kappa_pairs, law in laws, n in n_grid, rho in rho_grid
        p = rectangular_row_count(n)
        T1 = make_block_variance_profile(p, n, kappa1)
        T2 = make_block_variance_profile(p, n, kappa2)
        d1 = sqrt(2.0 * opnorm(T1, 2)) .* gamma1
        d2 = sqrt(2.0 * opnorm(T2, 2)) .* gamma2

        left_basis = block_spike_vectors(p, 4)
        right_basis = block_spike_vectors(n, 4)
        U1 = left_basis[:, 1:2]
        V1 = right_basis[:, 1:2]
        R_U = rho .* A_U ./ opnorm(A_U, 2)
        R_V = rho .* A_V ./ opnorm(A_V, 2)
        U2 = U1 * R_U +
             left_basis[:, 3:4] * sqrt(Symmetric(I - transpose(R_U) * R_U))
        V2 = V1 * R_V +
             right_basis[:, 3:4] * sqrt(Symmetric(I - transpose(R_V) * R_V))
        signal1 = make_signal_matrix(U1, d1, V1)
        signal2 = make_signal_matrix(U2, d2, V2)
        rho_true = rectangular_signal_correlation(d1, d2, R_U, R_V)
        orbit_errors = Float64[]
        correlation_errors = Float64[]
        fallback_count = 0
        nonreal_replications = 0
        total_mask_retries = 0
        first_nonreal_eigenvalue = nothing

        description =
            "Algorithm 4: kappas=($(kappa1), $(kappa2)), law=$(law), n=$(n), rho=$(rho)"
        @showprogress desc=description barglyphs=BarGlyphs(' ', '=', '>', ' ', ' ') for b in 1:B
            X1 = make_rectangular_noise(
                p,
                n;
                variance_profile=T1,
                law=law,
                rng=data_rng,
            )
            X2 = make_rectangular_noise(
                p,
                n;
                variance_profile=T2,
                law=law,
                rng=data_rng,
            )
            result = try
                algorithm4(
                    X1 + signal1,
                    X2 + signal2,
                    r;
                    rng=data_rng,
                    eigs_rng=eigs_rng,
                    return_details=true,
                )
            catch error_value
                error_value isa InterruptException && rethrow()
                if error_value isa Algorithm4MaskRetryError
                    total_mask_retries += error_value.total_mask_retries
                    nonreal_replications += 1
                    first_nonreal_eigenvalue === nothing &&
                        (first_nonreal_eigenvalue = error_value.eigenvalue)
                    continue
                end
                rethrow()
            end
            push!(
                orbit_errors,
                simultaneous_orbit_error(
                    result.R_U_eqv,
                    result.R_V_eqv,
                    R_U,
                    R_V,
                ),
            )
            push!(correlation_errors, abs(result.rho_sig - rho_true))
            fallback_count += result.sample1.used_fallback
            fallback_count += result.sample2.used_fallback
            total_mask_retries += result.sample1.mask_retries
            total_mask_retries += result.sample2.mask_retries
        end

        if fallback_count > 0
            warning_message =
                "Algorithm 4 used the dense fallback $(fallback_count) times " *
                "among valid replications"
            @warn warning_message kappa1 kappa2 law n rho
        end
        valid_replications = length(orbit_errors)
        if nonreal_replications > 0
            warning_message =
                "Algorithm 4 skipped replications whose selected outliers remained " *
                "non-real after $(ALGORITHM4_MAX_MASK_ATTEMPTS) mask attempts; " *
                "reported means use only valid replications"
            @warn warning_message kappa1 kappa2 law n rho valid_replications nonreal_replications total_mask_retries first_nonreal_eigenvalue
        end
        push!(
            rows,
            Any[
                kappa1,
                kappa2,
                string(law),
                n,
                p,
                rho,
                rho_true,
                isempty(orbit_errors) ? NaN : mean(orbit_errors),
                isempty(correlation_errors) ? NaN : mean(correlation_errors),
                valid_replications,
                nonreal_replications,
                total_mask_retries,
            ],
        )
    end

    return rows
end

function print_algorithm4_table(rows, value_column, title)
    isempty(rows) && return

    kappa_pairs = unique([(Int(row[1]), Int(row[2])) for row in rows])
    laws = unique([String(row[3]) for row in rows])
    n_grid = sort(unique([Int(row[4]) for row in rows]))
    rho_grid = sort(unique([Float64(row[6]) for row in rows]))
    results = Dict(
        (
            Int(row[1]),
            Int(row[2]),
            String(row[3]),
            Int(row[4]),
            Float64(row[6]),
        ) => Float64(row[value_column]) for row in rows
    )

    n_width = max(4, maximum(length.(string.(n_grid))))
    cell_width = 8
    group_width = length(rho_grid) * cell_width + 3 * (length(rho_grid) - 1)
    rho_header = join(
        [center_text(@sprintf("rho=%.2f", rho), cell_width) for rho in rho_grid],
        " | ",
    )
    profile_header = join(
        [
            center_text(
                "(T_rec^($(kappa1)), T_rec^($(kappa2)))",
                group_width,
            ) for (kappa1, kappa2) in kappa_pairs
        ],
        " || ",
    )
    separator = repeat("-", n_width) * "-+-" *
        join(fill(repeat("-", group_width), length(kappa_pairs)), "-++-")

    println("\n$(title)")
    println(repeat(" ", n_width) * " | " * profile_header)
    println(center_text("n", n_width) * " | " *
            join(fill(rho_header, length(kappa_pairs)), " || "))
    println(separator)

    for law in laws
        println(uppercasefirst(law))
        for n in n_grid
            groups = String[]
            for (kappa1, kappa2) in kappa_pairs
                cells = [
                    center_text(
                        @sprintf("%.4f", results[(kappa1, kappa2, law, n, rho)]),
                        cell_width,
                    ) for rho in rho_grid
                ]
                push!(groups, join(cells, " | "))
            end
            println(lpad(string(n), n_width) * " | " * join(groups, " || "))
        end
    end
end

function algorithm4_main()
    output_dir = joinpath(dirname(@__DIR__), "output", "csv")
    rows = simulate_algorithm4(
        [250, 500, 750, 1000],
        500,
        [(1, 1), (1, 4), (4, 4)],
        [:gaussian, :rademacher],
        [0.25, 0.60, 0.90],
        MersenneTwister(ALGORITHM4_BASE_SEED),
        MersenneTwister(ALGORITHM4_BASE_SEED + 1),
    )
    header = [
        "kappa_1",
        "kappa_2",
        "law",
        "n",
        "p",
        "rho",
        "rho_sig",
        "mean_joint_orbit_error_valid",
        "mean_signal_correlation_error_valid",
        "valid_replications",
        "nonreal_replications",
        "total_mask_retries",
    ]
    mkpath(output_dir)
    writedlm(
        joinpath(output_dir, "algorithm4.csv"),
        [permutedims(header); permutedims(reduce(hcat, rows))],
        ',',
    )

    print_algorithm4_table(
        rows,
        8,
        "Algorithm 4: mean joint common-sign orbit error over valid replications",
    )
    print_algorithm4_table(
        rows,
        9,
        "Algorithm 4: mean absolute signal-correlation error over valid replications",
    )
    println("Algorithm 4 results written to $(output_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    algorithm4_main()
end
