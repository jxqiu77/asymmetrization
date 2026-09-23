#=
Implementation and numerical simulation for Algorithm 3.

Usage:
  julia --project=. code/Algorithm3.jl
=#

using DelimitedFiles
using LinearAlgebra
using Printf
using ProgressMeter
using Random
using Statistics

isdefined(@__MODULE__, :ASYMMETRIZATION_EIGS_TOL) ||
    include(joinpath(@__DIR__, "asymmetrization_utils.jl"))

const ALGORITHM3_BASE_SEED = 20260813

"""
    rectangular_spike_estimates(Y, P, r, eigs_rng; solver=:auto)

Apply Algorithm 3 using a supplied mask. The nonzero eigenvalues of the split
dilation are ordered directly by real part, exactly as prescribed in the
algorithm.
"""
function rectangular_spike_estimates(
    Y::AbstractMatrix{<:Real},
    P::AbstractMatrix{Bool},
    r::Int,
    eigs_rng::AbstractRNG;
    solver::Symbol=:auto,
)
    p, n = validate_rectangular_algorithm_input(Y, r)
    size(P) == (p, n) || error("Y and P must have the same size")
    split_matrix = make_block_matrix(Y, P)
    result = extreme_eigenpairs(
        split_matrix,
        r,
        :LR,
        eigs_rng;
        solver=solver,
        ritzvec=false,
    )

    order = sortperm(real.(result.values); rev=true)
    positive_eigenvalues = result.values[order]
    estimates = 2.0 .* real.(positive_eigenvalues)
    return (
        estimates=estimates,
        positive_eigenvalues=positive_eigenvalues,
        used_fallback=result.used_fallback,
        fallback_reason=result.fallback_reason,
    )
end

"""
    algorithm3(Y, r; rng=Random.default_rng(), eigs_rng=rng, solver=:auto,
               return_details=false)

Draw one iid Bernoulli mask and estimate the `r` rectangular singular spikes.
"""
function algorithm3(
    Y::AbstractMatrix{<:Real},
    r::Int;
    rng::AbstractRNG=Random.default_rng(),
    eigs_rng::AbstractRNG=rng,
    solver::Symbol=:auto,
    return_details::Bool=false,
)
    p, n = validate_rectangular_algorithm_input(Y, r)
    P = make_rectangular_mask(p, n; rng=rng)
    result = rectangular_spike_estimates(Y, P, r, eigs_rng; solver=solver)
    if return_details
        return merge(result, (mask=P,))
    end
    return result.estimates
end

function simulate_algorithm3(n_grid, B, kappas, laws, data_rng, eigs_rng)
    rows = Vector{Vector{Any}}()
    gamma_grid = ([2.0, 1.6, 1.3], [2.0, 1.5, 1.5])
    r = 3

    for gamma in gamma_grid, kappa in kappas, law in laws, n in n_grid
        p = rectangular_row_count(n)
        T = make_block_variance_profile(p, n, kappa)
        d = sqrt(2.0 * opnorm(T, 2)) .* gamma
        U = block_spike_vectors(p, r)
        V = block_spike_vectors(n, r)
        signal = make_signal_matrix(U, d, V)
        estimates = Matrix{Float64}(undef, B, r)
        fallback_count = 0

        description = "Algorithm 3: gamma=$(gamma), kappa=$(kappa), law=$(law), n=$(n)"
        @showprogress desc=description barglyphs=BarGlyphs(' ', '=', '>', ' ', ' ') for b in 1:B
            X = make_rectangular_noise(
                p,
                n;
                variance_profile=T,
                law=law,
                rng=data_rng,
            )
            result = algorithm3(
                X + signal,
                r;
                rng=data_rng,
                eigs_rng=eigs_rng,
                return_details=true,
            )
            estimates[b, :] .= result.estimates
            fallback_count += result.used_fallback
        end

        if fallback_count > 0
            warning_message =
                "Algorithm 3 used the dense fallback $(fallback_count) times"
            @warn warning_message gamma kappa law n
        end

        for k in 1:r
            values = estimates[:, k]
            push!(
                rows,
                Any[
                    kappa,
                    string(law),
                    gamma[1],
                    gamma[2],
                    gamma[3],
                    n,
                    p,
                    k,
                    mean(abs.(values .- d[k])),
                    std(values),
                ],
            )
        end
    end

    return rows
end

function print_algorithm3_table(rows)
    isempty(rows) && return

    kappas = unique([Int(row[1]) for row in rows])
    laws = unique([String(row[2]) for row in rows])
    gamma_grid = unique([
        (Float64(row[3]), Float64(row[4]), Float64(row[5])) for row in rows
    ])
    n_grid = sort(unique([Int(row[6]) for row in rows]))
    components = sort(unique([Int(row[8]) for row in rows]))
    results = Dict(
        (
            Int(row[1]),
            String(row[2]),
            (Float64(row[3]), Float64(row[4]), Float64(row[5])),
            Int(row[6]),
            Int(row[8]),
        ) => (Float64(row[9]), Float64(row[10])) for row in rows
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
                "$(kappa == 1 ? "Homogeneous" : "Heterogeneous") T_rec^($(kappa))",
                group_width,
            ) for kappa in kappas
        ],
        " || ",
    )
    separator = repeat("-", n_width) * "-+-" *
        join(fill(repeat("-", group_width), length(kappas)), "-++-")

    println("\nAlgorithm 3: MAE (SD)")
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
                        push!(
                            cells,
                            center_text(
                                @sprintf("%.4f (%.4f)", mae, sd),
                                cell_width,
                            ),
                        )
                    end
                    push!(groups, join(cells, " | "))
                end
                println(lpad(string(n), n_width) * " | " * join(groups, " || "))
            end
        end
    end
end

function algorithm3_main()
    output_dir = joinpath(dirname(@__DIR__), "output", "csv")
    rows = simulate_algorithm3(
        [250, 500, 750, 1000],
        500,
        [1, 4],
        [:gaussian, :rademacher],
        MersenneTwister(ALGORITHM3_BASE_SEED),
        MersenneTwister(ALGORITHM3_BASE_SEED + 1),
    )
    header = [
        "kappa",
        "law",
        "gamma_1",
        "gamma_2",
        "gamma_3",
        "n",
        "p",
        "k",
        "mae_k",
        "sd_k",
    ]
    mkpath(output_dir)
    writedlm(
        joinpath(output_dir, "algorithm3.csv"),
        [permutedims(header); permutedims(reduce(hcat, rows))],
        ',',
    )

    print_algorithm3_table(rows)
    println("Algorithm 3 results written to $(output_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    algorithm3_main()
end
