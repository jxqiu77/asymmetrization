#=
Replication-level rank-one comparison for Algorithm 3 at n = 1000.

Usage:
  julia --project=. code/Algorithm3_Comparison.jl

Output:
  output/csv/algorithm3_comparison.csv
=#

using LinearAlgebra
using ProgressMeter
using Random

include(joinpath(@__DIR__, "Algorithm3.jl"))

const RECTANGULAR_COMPARISON_SEED = 20260813
const RECTANGULAR_COMPARISON_GAMMA = 1.5
const RECTANGULAR_COMPARISON_REPLICATIONS = 500
const RECTANGULAR_COMPARISON_KAPPAS = [1, 4, 8]
const RECTANGULAR_COMPARISON_LAWS = [:gaussian, :rademacher]
const RECTANGULAR_COMPARISON_N = 1000
const RECTANGULAR_COMPARISON_METHODS = (
    "Algorithm 3",
    "SN13",
    "OptShrink",
)
const RECTANGULAR_COMPARISON_OUTPUT = "algorithm3_comparison.csv"

function optshrink_d_rank_one(singular_values, n)
    p = length(singular_values)
    p >= 2 || return 0.0
    z = singular_values[1]
    z > 0.0 || return 0.0

    transform_sum = 0.0
    for singular_value in view(singular_values, 2:p)
        denominator = z^2 - singular_value^2
        denominator > 0.0 || return 0.0
        transform_sum += z / denominator
    end

    left_transform = transform_sum / (p - 1)
    companion_transform =
        (transform_sum + (n - p) / z) / (n - 1)
    d_transform = left_transform * companion_transform
    isfinite(d_transform) && d_transform > 0.0 || return 0.0
    return positive_finite_or_zero(inv(sqrt(d_transform)))
end

function rectangular_rank_one_baselines(Y)
    p, n = size(Y)
    p < n || error("the comparison requires p < n")
    p >= 2 || error("Y must have at least two rows")

    singular_values = svdvals(Y)
    leading = singular_values[1]

    sigma_squared = sum(abs2, Y) / p
    sigma = sqrt(max(sigma_squared, 0.0))
    aspect_ratio = p / n
    center = leading^2 - sigma_squared * (1.0 + aspect_ratio)
    sn13_estimate = if leading > sigma * (1.0 + sqrt(aspect_ratio))
        discriminant =
            max(center^2 - 4.0 * aspect_ratio * sigma_squared^2, 0.0)
        sqrt(0.5 * (center + sqrt(discriminant)))
    else
        0.0
    end

    return (
        positive_finite_or_zero(sn13_estimate),
        optshrink_d_rank_one(singular_values, n),
    )
end

function simulate_rectangular_rank_one_estimates(
    replications,
    kappas,
    laws,
    data_rng,
    eigs_rng,
)
    rows = Vector{Vector{Any}}()
    n = RECTANGULAR_COMPARISON_N
    p = rectangular_row_count(n)

    for kappa in kappas, law in laws
        T = make_rectangular_variance_profile(p, n, kappa)
        strength = sqrt(2.0 * variance_profile_opnorm(T)) *
                   RECTANGULAR_COMPARISON_GAMMA
        U = rectangular_spike_vectors(p, 1)
        V = rectangular_spike_vectors(n, 1)
        signal = make_rectangular_signal_matrix(U, [strength], V)
        fallback_count = 0

        description =
            "Rectangular rank-one estimates: kappa=$(kappa), law=$(law), n=$(n)"
        @showprogress desc=description barglyphs=BarGlyphs(' ', '=', '>', ' ', ' ') for replication in 1:replications
            noise = make_rectangular_noise(
                p,
                n;
                variance_profile=T,
                law=law,
                rng=data_rng,
            )
            Y = noise + signal
            proposed = algorithm3(
                Y,
                1;
                rng=data_rng,
                eigs_rng=eigs_rng,
                return_details=true,
            )
            sn13_estimate, optshrink_d =
                rectangular_rank_one_baselines(Y)
            estimates = (
                proposed.estimates[1],
                sn13_estimate,
                optshrink_d,
            )

            for (method, estimate) in zip(
                RECTANGULAR_COMPARISON_METHODS,
                estimates,
            )
                push!(
                    rows,
                    Any[
                        n,
                        p,
                        kappa,
                        string(law),
                        RECTANGULAR_COMPARISON_GAMMA,
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
            @warn "Algorithm 3 used the dense fallback" kappa law n fallback_count
    end

    return rows
end

function rectangular_comparison_main()
    rows = simulate_rectangular_rank_one_estimates(
        RECTANGULAR_COMPARISON_REPLICATIONS,
        RECTANGULAR_COMPARISON_KAPPAS,
        RECTANGULAR_COMPARISON_LAWS,
        MersenneTwister(RECTANGULAR_COMPARISON_SEED),
        MersenneTwister(RECTANGULAR_COMPARISON_SEED + 1),
    )

    expected_rows =
        RECTANGULAR_COMPARISON_REPLICATIONS *
        length(RECTANGULAR_COMPARISON_KAPPAS) *
        length(RECTANGULAR_COMPARISON_LAWS) *
        length(RECTANGULAR_COMPARISON_METHODS)
    length(rows) == expected_rows ||
        error("Expected $(expected_rows) estimate rows, got $(length(rows))")

    header = [
        "n",
        "p",
        "kappa",
        "law",
        "gamma",
        "strength",
        "replication",
        "method",
        "estimate",
    ]
    output_dir = joinpath(dirname(@__DIR__), "output", "csv")
    output_path =
        joinpath(output_dir, RECTANGULAR_COMPARISON_OUTPUT)
    mkpath(output_dir)
    write_csv_rows(output_path, header, rows)

    println()
    println("Saved $(length(rows)) replication-level estimates.")
    println("CSV written to $(abspath(output_path))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    rectangular_comparison_main()
end
