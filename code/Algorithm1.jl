#=
Implementation and numerical simulation for Algorithm 1.

Usage:
  julia --project=. code/Algorithm1.jl
=#

using DelimitedFiles
using LinearAlgebra
using Printf
using ProgressMeter
using Random
using Statistics

isdefined(@__MODULE__, :ASYMMETRIZATION_EIGS_TOL) ||
    include(joinpath(@__DIR__, "asymmetrization_utils.jl"))

const ALGORITHM1_BASE_SEED = 20260813
const BGS25_SEMICIRCLE_CENTRAL_SECOND_MOMENT = 0.10626920612194599

"""
    bgs25_noise_variance(ordered_eigenvalues)

Estimate the homogeneous Wigner noise variance using the middle half of the
ordered eigenvalues, as in Section 3.4 of Bykhovskaya, Gorin, and Sodin (2025).
"""
function bgs25_noise_variance(ordered_eigenvalues::AbstractVector{<:Real})
    n = length(ordered_eigenvalues)
    n >= 2 || error("at least two eigenvalues are required")
    first_middle = fld(n, 4) + 1
    last_middle = fld(3 * n, 4)
    middle_eigenvalues = view(ordered_eigenvalues, first_middle:last_middle)
    return sum(abs2, middle_eigenvalues) /
           (BGS25_SEMICIRCLE_CENTRAL_SECOND_MOMENT * n)
end

"""
    algorithm1(Y, r; rng=Random.default_rng(), eigs_rng=rng,
               solver=:auto, return_details=false)

Draw one symmetric Bernoulli mask and return `2real(lambda)` for the `r`
eigenvalues of the split matrix with largest real parts.
"""
function algorithm1(
    Y::AbstractMatrix{<:Real},
    r::Int;
    rng::AbstractRNG=Random.default_rng(),
    eigs_rng::AbstractRNG=rng,
    solver::Symbol=:auto,
    return_details::Bool=false,
)
    n = validate_wigner_algorithm_input(Y, r)
    P = make_symmetric_mask(n; rng=rng)
    split_matrix = make_block_matrix(Y, P)
    eigen_result = extreme_eigenpairs(
        split_matrix,
        r,
        :LR,
        eigs_rng;
        solver=solver,
        ritzvec=false,
    )
    order = sortperm(real.(eigen_result.values); rev=true)
    positive_eigenvalues = eigen_result.values[order]
    result = (
        estimates=2.0 .* real.(positive_eigenvalues),
        positive_eigenvalues=positive_eigenvalues,
        mask=P,
        used_fallback=eigen_result.used_fallback,
        fallback_reason=eigen_result.fallback_reason,
    )
    return return_details ? result : result.estimates
end

function simulate_algorithm1(n_grid, B, kappas, laws, rng, eigs_rng)
    rows = Vector{Vector{Any}}()
    gamma_grid = ([2.0, 1.6, 1.3], [2.0, 1.5, 1.5])
    r = 3

    for gamma in gamma_grid, kappa in kappas, law in laws, n in n_grid
        T = make_block_variance_profile(n, n, kappa)
        d = sqrt(2.0 * opnorm(T, 2)) .* gamma
        U = block_spike_vectors(n, r)
        signal = make_signal_matrix(U, d)
        estimates = Matrix{Float64}(undef, B, r)
        fallback_count = 0

        description = "Algorithm 1: gamma=$(gamma), kappa=$(kappa), law=$(law), n=$(n)"
        @showprogress desc=description barglyphs=BarGlyphs(' ', '=', '>', ' ', ' ') for b in 1:B
            X = make_wigner(n; variance_profile=T, law=law, rng=rng)
            result = algorithm1(
                X + signal,
                r;
                rng=rng,
                eigs_rng=eigs_rng,
                return_details=true,
            )
            estimates[b, :] .= result.estimates
            fallback_count += result.used_fallback
        end

        if fallback_count > 0
            @warn "Algorithm 1 used the dense fallback $(fallback_count) times for gamma=$(gamma), kappa=$(kappa), law=$(law), n=$(n)"
        end

        for k in 1:r
            values = estimates[:, k]
            mae = mean(abs.(values .- d[k]))
            empirical_sd = std(values)
            push!(
                rows,
                Any[
                    kappa,
                    string(law),
                    gamma[1],
                    gamma[2],
                    gamma[3],
                    n,
                    k,
                    mae,
                    empirical_sd,
                ],
            )
        end
    end

    return rows
end

function print_algorithm1_table(rows)
    isempty(rows) && return

    kappas = unique([Int(row[1]) for row in rows])
    laws = unique([String(row[2]) for row in rows])
    gamma_grid = unique([
        (Float64(row[3]), Float64(row[4]), Float64(row[5])) for row in rows
    ])
    n_grid = sort(unique([Int(row[6]) for row in rows]))
    components = sort(unique([Int(row[7]) for row in rows]))
    results = Dict(
        (
            Int(row[1]),
            String(row[2]),
            (Float64(row[3]), Float64(row[4]), Float64(row[5])),
            Int(row[6]),
            Int(row[7]),
        ) => (Float64(row[8]), Float64(row[9])) for row in rows
    )

    n_width = max(4, maximum(length.(string.(n_grid))))
    cell_width = 15
    group_width = length(components) * cell_width + 3 * (length(components) - 1)
    component_header = join(
        [center_text("k=$(k)", cell_width) for k in components],
        " | ",
    )
    profile_header = join(
        [
            center_text(
                "$(kappa == 1 ? "Homogeneous" : "Heterogeneous") T^($(kappa))",
                group_width,
            ) for kappa in kappas
        ],
        " || ",
    )
    separator = repeat("-", n_width) * "-+-" *
        join(fill(repeat("-", group_width), length(kappas)), "-++-")

    println("\nAlgorithm 1: MAE (SD)")
    for gamma in gamma_grid
        println("\ngamma = $(@sprintf("(%.2f, %.2f, %.2f)", gamma...))")
        println(repeat(" ", n_width) * " | " * profile_header)
        println(center_text("n", n_width) * " | " *
                join(fill(component_header, length(kappas)), " || "))
        println(separator)

        for law in laws
            println(uppercasefirst(law))
            for n in n_grid
                groups = String[]
                for kappa in kappas
                    cells = String[]
                    for k in components
                        mae, sd = results[(kappa, law, gamma, n, k)]
                        push!(cells, center_text(@sprintf("%.4f (%.4f)", mae, sd), cell_width))
                    end
                    push!(groups, join(cells, " | "))
                end
                println(lpad(string(n), n_width) * " | " * join(groups, " || "))
            end
        end
    end
end

function algorithm1_main()
    output_dir = joinpath(dirname(@__DIR__), "output", "csv")
    rows = simulate_algorithm1(
        [250, 500, 750, 1000],
        500,
        [1, 4],
        [:gaussian, :rademacher],
        MersenneTwister(ALGORITHM1_BASE_SEED),
        MersenneTwister(ALGORITHM1_BASE_SEED + 1),
    )
    header = [
        "kappa",
        "law",
        "gamma_1",
        "gamma_2",
        "gamma_3",
        "n",
        "k",
        "mae_k",
        "sd_k",
    ]
    mkpath(output_dir)
    writedlm(
        joinpath(output_dir, "algorithm1.csv"),
        [permutedims(header); permutedims(reduce(hcat, rows))],
        ',',
    )

    print_algorithm1_table(rows)
    println("Algorithm 1 results written to $(output_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    algorithm1_main()
end
