#=
Replication-level rank-one comparison for Algorithm 1 at n = 1000.

Usage:
  julia --project=. code/Algorithm1_Comparison.jl

Output:
  output/csv/algorithm1_comparison.csv
=#

using LinearAlgebra
using ProgressMeter
using Random
using Statistics

include(joinpath(@__DIR__, "Algorithm1.jl"))

const WIGNER_COMPARISON_SEED = 20260813
const WIGNER_COMPARISON_GAMMA = 1.5
const WIGNER_COMPARISON_REPLICATIONS = 500
const WIGNER_COMPARISON_KAPPAS = [1, 4, 8]
const WIGNER_COMPARISON_LAWS = [:gaussian, :rademacher]
const WIGNER_COMPARISON_METHODS = ("Algorithm 1", "BGS25", "BGN11")
const WIGNER_COMPARISON_N = 1000
const WIGNER_COMPARISON_OUTPUT = "algorithm1_comparison.csv"

function wigner_rank_one_baselines(Y)
    n, m = size(Y)
    n == m || error("Y must be square")
    n >= 2 || error("Y must have at least two rows")
    issymmetric(Y) || error("Y must be symmetric")

    eigenvalues = sort!(eigvals(Symmetric(Y)); rev=true)
    leading = eigenvalues[1]
    sigma_squared = bgs25_noise_variance(eigenvalues)
    sigma = sqrt(max(sigma_squared, 0.0))
    inverse_bbp = if leading > 2.0 * sigma
        discriminant = max(leading^2 - 4.0 * sigma_squared, 0.0)
        0.5 * (leading + sqrt(discriminant))
    else
        0.0
    end
    inverse_bbp = positive_finite_or_zero(inverse_bbp)

    empirical_transform =
        mean(1.0 ./ (view(eigenvalues, 2:n) .- leading))
    empirical_m = positive_finite_or_zero(-inv(empirical_transform))

    return inverse_bbp, empirical_m
end

function simulate_wigner_rank_one_estimates(
    replications,
    kappas,
    laws,
    data_rng,
    eigs_rng,
)
    rows = Vector{Vector{Any}}()
    n = WIGNER_COMPARISON_N

    for kappa in kappas, law in laws
        profile = make_block_variance_profile(n, n, kappa)
        strength = sqrt(2.0 * variance_profile_opnorm(profile)) *
                   WIGNER_COMPARISON_GAMMA
        direction = block_spike_vectors(n, 1)
        signal = make_signal_matrix(direction, [strength])
        fallback_count = 0

        description =
            "Wigner rank-one estimates: kappa=$(kappa), law=$(law), n=$(n)"
        @showprogress desc=description barglyphs=BarGlyphs(' ', '=', '>', ' ', ' ') for replication in 1:replications
            noise = make_wigner(
                n;
                variance_profile=profile,
                law=law,
                rng=data_rng,
            )
            Y = noise + signal
            proposed = algorithm1(
                Y,
                1;
                rng=data_rng,
                eigs_rng=eigs_rng,
                return_details=true,
            )
            inverse_bbp, empirical_m = wigner_rank_one_baselines(Y)
            estimates = (
                proposed.estimates[1],
                inverse_bbp,
                empirical_m,
            )

            for (method, estimate) in zip(WIGNER_COMPARISON_METHODS, estimates)
                push!(
                    rows,
                    Any[
                        n,
                        kappa,
                        string(law),
                        WIGNER_COMPARISON_GAMMA,
                        strength,
                        replication,
                        method,
                        estimate,
                    ],
                )
            end
            fallback_count += proposed.used_fallback
        end

        fallback_count > 0 &&
            @warn "Algorithm 1 used the dense fallback" kappa law n fallback_count
    end

    return rows
end

function wigner_comparison_main()
    rows = simulate_wigner_rank_one_estimates(
        WIGNER_COMPARISON_REPLICATIONS,
        WIGNER_COMPARISON_KAPPAS,
        WIGNER_COMPARISON_LAWS,
        MersenneTwister(WIGNER_COMPARISON_SEED),
        MersenneTwister(WIGNER_COMPARISON_SEED + 1),
    )

    expected_rows =
        WIGNER_COMPARISON_REPLICATIONS *
        length(WIGNER_COMPARISON_KAPPAS) *
        length(WIGNER_COMPARISON_LAWS) *
        length(WIGNER_COMPARISON_METHODS)
    length(rows) == expected_rows ||
        error("Expected $(expected_rows) estimate rows, got $(length(rows))")

    header = [
        "n",
        "kappa",
        "law",
        "gamma",
        "strength",
        "replication",
        "method",
        "estimate",
    ]
    output_dir = joinpath(dirname(@__DIR__), "output", "csv")
    output_path = joinpath(output_dir, WIGNER_COMPARISON_OUTPUT)
    mkpath(output_dir)
    write_csv_rows(output_path, header, rows)

    println()
    println("Saved $(length(rows)) replication-level estimates.")
    println("CSV written to $(abspath(output_path))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    wigner_comparison_main()
end
