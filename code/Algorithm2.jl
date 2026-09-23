#=
Implementation and numerical simulation for Algorithm 2.

Usage:
  julia --project=. code/Algorithm2.jl
=#

using DelimitedFiles
using LinearAlgebra
using Printf
using ProgressMeter
using Random
using Statistics

isdefined(@__MODULE__, :ASYMMETRIZATION_EIGS_TOL) ||
    include(joinpath(@__DIR__, "asymmetrization_utils.jl"))

const ALGORITHM2_BASE_SEED = 20260813

function orbit_error(estimate, target)
    error = Inf
    for a in (-1.0, 1.0), b in (-1.0, 1.0), c in (-1.0, 1.0), d in (-1.0, 1.0)
        signed_target = Diagonal([a, b]) * target * Diagonal([c, d])
        error = min(error, maximum(abs.(estimate .- signed_target)))
    end
    return error
end

"""
    wigner_signal_correlation(d1, d2, R)

Compute the normalized correlation between two symmetric signals from their
spike strengths and overlap matrix.
"""
function wigner_signal_correlation(d1, d2, R)
    size(R) == (length(d1), length(d2)) || error("R has incompatible dimensions")
    denominator = norm(d1) * norm(d2)
    denominator > 0.0 || error("spike-strength vectors must be nonzero")
    weights = d1 * transpose(d2)
    return sum(weights .* abs2.(R)) / denominator
end

function _one_sample_algorithm2(Y, r, data_rng, eigs_rng; solver)
    n = validate_wigner_algorithm_input(Y, r)
    P = make_symmetric_mask(n; rng=data_rng)
    factors = split_outlier_factors(
        make_block_matrix(Y, P),
        n,
        r,
        eigs_rng;
        solver=solver,
    )
    return merge(factors, (mask=P,))
end

"""
    algorithm2(Y1, Y2, r; rng=Random.default_rng(), eigs_rng=rng,
               solver=:auto, return_details=false)

Estimate the Wigner overlap matrix up to row/column signs and the normalized
signal correlation. Each sample uses one independently drawn symmetric mask.
"""
function algorithm2(
    Y1::AbstractMatrix{<:Real},
    Y2::AbstractMatrix{<:Real},
    r::Int;
    rng::AbstractRNG=Random.default_rng(),
    eigs_rng::AbstractRNG=rng,
    solver::Symbol=:auto,
    return_details::Bool=false,
)
    size(Y1) == size(Y2) || error("Y1 and Y2 must have the same size")
    validate_wigner_algorithm_input(Y1, r)
    validate_wigner_algorithm_input(Y2, r)
    sample1 = _one_sample_algorithm2(Y1, r, rng, eigs_rng; solver=solver)
    sample2 = _one_sample_algorithm2(Y2, r, rng, eigs_rng; solver=solver)

    R_abs = zeros(Float64, r, r)
    R_eqv = zeros(Float64, r, r)
    for i in 1:r, j in 1:r
        cross_12 = dot(sample1.left_first[:, i], sample2.right_first[:, j])
        cross_21 = dot(sample2.left_first[:, j], sample1.right_first[:, i])
        q = 4.0 * cross_12 * cross_21
        R_abs[i, j] = sqrt(clamp(q, 0.0, 1.0))
        R_eqv[i, j] = (cross_12 >= 0.0 ? 1.0 : -1.0) * R_abs[i, j]
    end

    rho_sig = wigner_signal_correlation(
        sample1.estimates,
        sample2.estimates,
        R_eqv,
    )
    result = (R_eqv=R_eqv, rho_sig=rho_sig)
    return return_details ?
           merge(result, (R_abs=R_abs, sample1=sample1, sample2=sample2)) :
           result
end

function simulate_algorithm2(n_grid, B, kappa_pairs, laws, rho_grid, data_rng, eigs_rng)
    rows = Vector{Vector{Any}}()
    r = 2
    R0 = [0.8 0.3; -0.2 0.6]

    for (kappa1, kappa2) in kappa_pairs, law in laws, n in n_grid, rho in rho_grid
        T1 = make_block_variance_profile(n, n, kappa1)
        T2 = make_block_variance_profile(n, n, kappa2)
        directions = block_spike_vectors(n, 4)
        U1 = directions[:, 1:2]
        R = rho .* R0 ./ opnorm(R0, 2)
        U2 = U1 * R + directions[:, 3:4] * sqrt(Symmetric(I - R' * R))
        d1 = [5.0, 4.0]
        d2 = [6.0, 4.5]
        signal1 = make_signal_matrix(U1, d1)
        signal2 = make_signal_matrix(U2, d2)
        rho_sig = wigner_signal_correlation(d1, d2, R)
        orbit_errors = zeros(B)
        correlation_errors = zeros(B)
        fallback_count = 0

        description = "Algorithm 2: kappas=($(kappa1), $(kappa2)), law=$(law), n=$(n), rho=$(rho)"
        @showprogress desc=description barglyphs=BarGlyphs(' ', '=', '>', ' ', ' ') for b in 1:B
            W1 = make_wigner(n; variance_profile=T1, law=law, rng=data_rng)
            W2 = make_wigner(n; variance_profile=T2, law=law, rng=data_rng)
            result = algorithm2(
                W1 + signal1,
                W2 + signal2,
                r;
                rng=data_rng,
                eigs_rng=eigs_rng,
                return_details=true,
            )
            orbit_errors[b] = orbit_error(result.R_eqv, R)
            correlation_errors[b] = abs(result.rho_sig - rho_sig)
            fallback_count += result.sample1.used_fallback
            fallback_count += result.sample2.used_fallback
        end

        if fallback_count > 0
            @warn "Algorithm 2 used the dense fallback $(fallback_count) times" kappa1 kappa2 law n rho
        end

        push!(
            rows,
            Any[
                kappa1,
                kappa2,
                string(law),
                n,
                rho,
                rho_sig,
                mean(orbit_errors),
                mean(correlation_errors),
            ],
        )
    end

    return rows
end

function print_algorithm2_table(rows, value_column, title)
    isempty(rows) && return

    kappa_pairs = unique([(Int(row[1]), Int(row[2])) for row in rows])
    laws = unique([String(row[3]) for row in rows])
    n_grid = sort(unique([Int(row[4]) for row in rows]))
    rho_grid = sort(unique([Float64(row[5]) for row in rows]))
    results = Dict(
        (
            Int(row[1]),
            Int(row[2]),
            String(row[3]),
            Int(row[4]),
            Float64(row[5]),
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
            center_text("(T^($(kappa1)), T^($(kappa2)))", group_width) for
            (kappa1, kappa2) in kappa_pairs
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

function algorithm2_main()
    output_dir = joinpath(dirname(@__DIR__), "output", "csv")
    rows = simulate_algorithm2(
        [250, 500, 750, 1000],
        500,
        [(1, 1), (1, 4), (4, 4)],
        [:gaussian, :rademacher],
        [0.25, 0.60, 0.90],
        MersenneTwister(ALGORITHM2_BASE_SEED),
        MersenneTwister(ALGORITHM2_BASE_SEED + 1),
    )
    header = [
        "kappa_1",
        "kappa_2",
        "law",
        "n",
        "rho",
        "rho_sig",
        "mean_orbit_error",
        "mean_signal_correlation_error",
    ]
    mkpath(output_dir)
    writedlm(
        joinpath(output_dir, "algorithm2.csv"),
        [permutedims(header); permutedims(reduce(hcat, rows))],
        ',',
    )

    print_algorithm2_table(rows, 7, "Algorithm 2: mean sign-equivalence-class error")
    print_algorithm2_table(rows, 8, "Algorithm 2: mean absolute signal-correlation error")
    println("Algorithm 2 results written to $(output_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    algorithm2_main()
end
