#=
Generate the Wigner and rectangular distinct/repeated-spike figures.

Both cases are written as separate two-panel figures for the kappa=1 and
kappa=4 variance profiles.

Usage:
  julia --project=. code/distinct_repeated_spike.jl
=#

using CairoMakie
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "asymmetrization_utils.jl"))

const DISTINCT_GAMMAS = [2.0, 1.6, 1.3]
const REPEATED_GAMMAS = [2.0, 1.5, 1.5]
const KAPPAS = (1, 4)
const DISTINCT_SEED = 20260814
const REPEATED_SEED = 20260814
const DEFAULT_WIGNER_OUTPUTS = (
    joinpath(
        dirname(@__DIR__),
        "output",
        "figure",
        "distinct_repeated_spike_wigner_homogeneous.pdf",
    ),
    joinpath(
        dirname(@__DIR__),
        "output",
        "figure",
        "distinct_repeated_spike_wigner_heterogeneous.pdf",
    ),
)
const DEFAULT_OUTPUTS = (
    joinpath(dirname(@__DIR__), "output", "figure", "distinct_repeated_spike_rectangular_homogeneous.pdf"),
    joinpath(dirname(@__DIR__), "output", "figure", "distinct_repeated_spike_rectangular_heterogeneous.pdf"),
)

function rectangular_block_eigenvalues(X, P)
    size(X) == size(P) || error("X and P must have the same size")
    p, n = size(X)
    p <= n || error("the rectangular figure requires p <= n")
    A = P .* X
    B = (1.0 .- P) .* X
    roots = sqrt.(complex.(eigvals!(A * transpose(B))))
    return vcat(roots, -roots, zeros(ComplexF64, n - p))
end

function wigner_realization(n, T, T_norm, gammas, seed)
    U = block_spike_vectors(n, length(gammas))
    identity_matrix = Matrix{Float64}(I, length(gammas), length(gammas))
    @assert isapprox(U' * U, identity_matrix; atol=1.0e-12)

    strengths = sqrt(2.0 * T_norm) .* gammas
    signal = make_signal_matrix(U, strengths)
    rng = MersenneTwister(seed)
    W = make_wigner(n; variance_profile=T, law=:gaussian, rng=rng)
    P = make_symmetric_mask(n; rng=rng)
    eigenvalues = block_eigenvalues(W + signal, P)
    @assert length(eigenvalues) == 2 * n

    targets = sort(vcat(-strengths ./ 2.0, strengths ./ 2.0))
    return eigenvalues, unique(targets), targets, sqrt(T_norm / 2.0)
end

function rectangular_realization(T, T_norm, gammas, seed)
    p, n = size(T)
    U = block_spike_vectors(p, length(gammas))
    V = block_spike_vectors(n, length(gammas))
    identity_matrix = Matrix{Float64}(I, length(gammas), length(gammas))
    @assert isapprox(U' * U, identity_matrix; atol=1.0e-12)
    @assert isapprox(V' * V, identity_matrix; atol=1.0e-12)

    strengths = sqrt(2.0 * T_norm) .* gammas
    signal = Matrix(U * Diagonal(strengths) * V')
    rng = MersenneTwister(seed)
    X = sqrt.(T) .* randn(rng, p, n)
    P = Float64.(rand(rng, p, n) .<= 0.5)
    eigenvalues = rectangular_block_eigenvalues(X + signal, P)
    @assert length(eigenvalues) == p + n

    targets = sort(vcat(-strengths ./ 2.0, strengths ./ 2.0))
    return eigenvalues, unique(targets), targets, sqrt(T_norm / 2.0)
end

function match_outliers(eigenvalues, targets)
    target_count = length(targets)
    state_count = 1 << target_count
    costs = fill(Inf, state_count)
    costs[1] = 0.0
    parent_state = fill(-1, length(eigenvalues), state_count)
    parent_target = fill(0, length(eigenvalues), state_count)

    for eigenvalue_index in eachindex(eigenvalues)
        next_costs = fill(Inf, state_count)
        for state in 0:(state_count - 1)
            current_cost = costs[state + 1]
            isfinite(current_cost) || continue

            if current_cost < next_costs[state + 1]
                next_costs[state + 1] = current_cost
                parent_state[eigenvalue_index, state + 1] = state
                parent_target[eigenvalue_index, state + 1] = 0
            end

            for target_index in eachindex(targets)
                bit = 1 << (target_index - 1)
                state & bit == 0 || continue
                next_state = state | bit
                candidate = current_cost + abs2(
                    eigenvalues[eigenvalue_index] - targets[target_index],
                )
                if candidate < next_costs[next_state + 1]
                    next_costs[next_state + 1] = candidate
                    parent_state[eigenvalue_index, next_state + 1] = state
                    parent_target[eigenvalue_index, next_state + 1] = target_index
                end
            end
        end
        costs = next_costs
    end

    matched_indices = fill(0, target_count)
    state = state_count - 1
    for eigenvalue_index in length(eigenvalues):-1:1
        previous_state = parent_state[eigenvalue_index, state + 1]
        previous_state >= 0 || error("failed to match outlier eigenvalues")
        target_index = parent_target[eigenvalue_index, state + 1]
        if target_index > 0
            matched_indices[target_index] = eigenvalue_index
        end
        state = previous_state
    end

    return eigenvalues[matched_indices]
end

function draw_panel!(axis, eigenvalues, unique_targets, targets, edge, x_limits, y_limits; panel_label)
    scatter!(
        axis,
        real.(eigenvalues),
        imag.(eigenvalues);
        color=(:steelblue, 0.40),
        markersize=6.0,
        strokewidth=0,
    )

    theta = range(0.0, 2.0 * pi; length=400)
    lines!(
        axis,
        edge .* cos.(theta),
        edge .* sin.(theta);
        color=:black,
        linestyle=:dot,
        linewidth=2.0,
    )

    scatter!(
        axis,
        unique_targets,
        zeros(length(unique_targets));
        color=:red,
        marker=:star5,
        markersize=18.0,
        strokewidth=0,
    )

    matched = match_outliers(eigenvalues, targets)
    scatter!(
        axis,
        real.(matched),
        imag.(matched);
        color=:blue,
        markersize=9.0,
        strokewidth=0,
    )

    xlims!(axis, x_limits...)
    ylims!(axis, y_limits...)
    text!(
        axis, 0.02, 0.98;
        text=panel_label,
        space=:relative,
        align=(:left, :top),
        font=:bold,
        fontsize=36,
        color=:black,
    )
end

function generate_wigner_figure3(; n=1000, output_paths=DEFAULT_WIGNER_OUTPUTS)
    n >= 3 || error("n must be at least 3")
    iseven(n) || error("n must be even")
    length(output_paths) == 2 || error("output_paths must contain two paths")

    gamma_sets = [DISTINCT_GAMMAS, REPEATED_GAMMAS]
    seeds = [DISTINCT_SEED, REPEATED_SEED]
    gamma_tuple = "(\u03b3\u2081, \u03b3\u2082, \u03b3\u2083)"
    panel_titles = [
        "Distinct spikes: $(gamma_tuple) = (2.00, 1.60, 1.30)",
        "Repeated spikes: $(gamma_tuple) = (2.00, 1.50, 1.50)",
    ]

    eigenvalues = Matrix{Vector{ComplexF64}}(undef, 2, 2)
    unique_targets = Matrix{Vector{Float64}}(undef, 2, 2)
    targets = Matrix{Vector{Float64}}(undef, 2, 2)
    edges = zeros(2, 2)

    for column in 1:2
        T = make_block_variance_profile(n, n, KAPPAS[column])
        @assert issymmetric(T)
        @assert minimum(T) > 0.0
        T_norm = variance_profile_opnorm(T)

        for row in 1:2
            eigenvalues[row, column],
            unique_targets[row, column],
            targets[row, column],
            edges[row, column] = wigner_realization(
                n,
                T,
                T_norm,
                gamma_sets[row],
                seeds[row],
            )
        end
    end

    @assert length(unique_targets[1, 1]) == 6
    @assert length(unique_targets[2, 1]) == 4
    @assert maximum(count(==(target), targets[2, 1]) for target in unique_targets[2, 1]) == 2

    x_values = Float64[]
    y_values = Float64[]
    for column in 1:2, row in 1:2
        append!(x_values, real.(eigenvalues[row, column]))
        append!(x_values, unique_targets[row, column])
        append!(x_values, [-edges[row, column], edges[row, column]])
        append!(y_values, imag.(eigenvalues[row, column]))
        append!(y_values, [-edges[row, column], edges[row, column]])
    end
    x_limits = padded_limits(x_values)
    y_limits = padded_limits(y_values)

    palatino_theme = Theme(
        fonts=Attributes(
            :regular => "Palatino",
            :bold => "Palatino Bold",
            :italic => "Palatino Italic",
            :bolditalic => "Palatino Bold Italic",
        ),
    )
    figures = with_theme(palatino_theme) do
        [begin
            fig = Figure(size=(850, 880), fontsize=22, figure_padding=24)
            for row in 1:2
                axis = Axis(
                    fig[row, 1];
                    title=panel_titles[row],
                    titlefont=:regular,
                    titlesize=22,
                    titlegap=8,
                    xlabel="Re",
                    ylabel="Im",
                    xlabelsize=22,
                    ylabelsize=22,
                    xticklabelsize=18,
                    yticklabelsize=18,
                    xgridcolor=(:gray, 0.18),
                    ygridcolor=(:gray, 0.18),
                    aspect=DataAspect(),
                )
                hidespines!(axis, :t, :r)
                draw_panel!(
                    axis,
                    eigenvalues[row, column],
                    unique_targets[row, column],
                    targets[row, column],
                    edges[row, column],
                    x_limits,
                    y_limits;
                    panel_label=string('A' + 2 * (row - 1) + column - 1),
                )
            end
            rowgap!(fig.layout, 12)
            fig
        end for column in 1:2]
    end

    for (output_path, figure) in zip(output_paths, figures)
        mkpath(dirname(output_path))
        save(output_path, figure)
        println("Wigner distinct/repeated-spike figure written to $(abspath(output_path))")
    end
    return output_paths
end

function generate_rectangular_figure3(; n=1000, output_paths=DEFAULT_OUTPUTS)
    n >= 10 || error("n must be at least 10")
    n % 10 == 0 || error("n must be divisible by 10")
    length(output_paths) == 2 || error("output_paths must contain two paths")
    p = 3 * n ÷ 5

    gamma_sets = [DISTINCT_GAMMAS, REPEATED_GAMMAS]
    seeds = [DISTINCT_SEED, REPEATED_SEED]
    gamma_tuple = "(\u03b3\u2081, \u03b3\u2082, \u03b3\u2083)"
    panel_titles = [
        "Distinct spikes: $(gamma_tuple) = (2.00, 1.60, 1.30)",
        "Repeated spikes: $(gamma_tuple) = (2.00, 1.50, 1.50)",
    ]

    eigenvalues = Matrix{Vector{ComplexF64}}(undef, 2, 2)
    unique_targets = Matrix{Vector{Float64}}(undef, 2, 2)
    targets = Matrix{Vector{Float64}}(undef, 2, 2)
    edges = zeros(2, 2)

    for column in 1:2
        T = make_block_variance_profile(p, n, KAPPAS[column])
        @assert size(T) == (p, n)
        @assert minimum(T) > 0.0
        T_norm = variance_profile_opnorm(T)

        for row in 1:2
            eigenvalues[row, column],
            unique_targets[row, column],
            targets[row, column],
            edges[row, column] = rectangular_realization(
                T,
                T_norm,
                gamma_sets[row],
                seeds[row],
            )
        end
    end

    @assert length(unique_targets[1, 1]) == 6
    @assert length(unique_targets[2, 1]) == 4
    @assert maximum(count(==(target), targets[2, 1]) for target in unique_targets[2, 1]) == 2

    x_values = Float64[]
    y_values = Float64[]
    for column in 1:2, row in 1:2
        append!(x_values, real.(eigenvalues[row, column]))
        append!(x_values, unique_targets[row, column])
        append!(x_values, [-edges[row, column], edges[row, column]])
        append!(y_values, imag.(eigenvalues[row, column]))
        append!(y_values, [-edges[row, column], edges[row, column]])
    end
    x_limits = padded_limits(x_values)
    y_limits = padded_limits(y_values)

    palatino_theme = Theme(
        fonts=Attributes(
            :regular => "Palatino",
            :bold => "Palatino Bold",
            :italic => "Palatino Italic",
            :bolditalic => "Palatino Bold Italic",
        ),
    )
    figures = with_theme(palatino_theme) do
        [begin
            fig = Figure(size=(850, 880), fontsize=22, figure_padding=24)
            for row in 1:2
                axis = Axis(
                    fig[row, 1];
                    title=panel_titles[row],
                    titlefont=:regular,
                    titlesize=22,
                    titlegap=8,
                    xlabel="Re",
                    ylabel="Im",
                    xlabelsize=22,
                    ylabelsize=22,
                    xticklabelsize=18,
                    yticklabelsize=18,
                    xgridcolor=(:gray, 0.18),
                    ygridcolor=(:gray, 0.18),
                    aspect=DataAspect(),
                )
                hidespines!(axis, :t, :r)
                draw_panel!(
                    axis,
                    eigenvalues[row, column],
                    unique_targets[row, column],
                    targets[row, column],
                    edges[row, column],
                    x_limits,
                    y_limits;
                    panel_label=string('A' + 2 * (row - 1) + column - 1),
                )
            end
            rowgap!(fig.layout, 12)
            fig
        end for column in 1:2]
    end

    for (output_path, figure) in zip(output_paths, figures)
        mkpath(dirname(output_path))
        save(output_path, figure)
        println("Distinct/repeated-spike figure written to $(abspath(output_path))")
    end
    return output_paths
end

function generate_figure3(
    ;
    n=1000,
    wigner_output_paths=DEFAULT_WIGNER_OUTPUTS,
    output_paths=DEFAULT_OUTPUTS,
)
    n >= 10 || error("n must be at least 10")
    n % 10 == 0 || error("n must be divisible by 10")
    wigner_outputs = generate_wigner_figure3(; n=n, output_paths=wigner_output_paths)
    rectangular_outputs = generate_rectangular_figure3(; n=n, output_paths=output_paths)
    return (wigner=wigner_outputs, rectangular=rectangular_outputs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_figure3()
end
