#=
Generate the Wigner and rectangular single-spike phase-transition figures.

Both cases are written as separate three-panel figures for the kappa=1 and
kappa=4 variance profiles.

Usage:
  julia --project=. code/phase_transition.jl
=#

using CairoMakie
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "asymmetrization_utils.jl"))

const GAMMAS = [0.80, 1.20, 1.60]
const KAPPAS = (1, 4)
const PANEL_SEEDS = [2026, 2027, 2028]
const DEFAULT_WIGNER_OUTPUTS = (
    joinpath(dirname(@__DIR__), "output", "figure", "phase_transition_wigner_homogeneous.pdf"),
    joinpath(dirname(@__DIR__), "output", "figure", "phase_transition_wigner_heterogeneous.pdf"),
)
const DEFAULT_OUTPUTS = (
    joinpath(dirname(@__DIR__), "output", "figure", "phase_transition_rectangular_homogeneous.pdf"),
    joinpath(dirname(@__DIR__), "output", "figure", "phase_transition_rectangular_heterogeneous.pdf"),
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

function wigner_single_spike_realization(n, T, T_norm, gamma, seed)
    U = block_spike_vectors(n, 1)
    @assert isapprox(norm(U[:, 1]), 1.0; atol=1.0e-12)

    edge = sqrt(T_norm / 2.0)
    strength = 2.0 * edge * gamma
    signal = make_signal_matrix(U, [strength])
    rng = MersenneTwister(seed)
    W = make_wigner(n; variance_profile=T, law=:gaussian, rng=rng)
    P = make_symmetric_mask(n; rng=rng)
    eigenvalues = block_eigenvalues(W + signal, P)
    @assert length(eigenvalues) == 2 * n

    return eigenvalues, [-strength / 2.0, strength / 2.0], edge
end

function rectangular_single_spike_realization(T, T_norm, gamma, seed)
    p, n = size(T)
    U = block_spike_vectors(p, 1)
    V = block_spike_vectors(n, 1)
    @assert isapprox(norm(U[:, 1]), 1.0; atol=1.0e-12)
    @assert isapprox(norm(V[:, 1]), 1.0; atol=1.0e-12)

    edge = sqrt(T_norm / 2.0)
    strength = 2.0 * edge * gamma
    signal = Matrix(U * Diagonal([strength]) * V')
    rng = MersenneTwister(seed)
    X = sqrt.(T) .* randn(rng, p, n)
    P = Float64.(rand(rng, p, n) .<= 0.5)
    eigenvalues = rectangular_block_eigenvalues(X + signal, P)
    @assert length(eigenvalues) == p + n

    return eigenvalues, [-strength / 2.0, strength / 2.0], edge
end

function draw_panel!(axis, eigenvalues, edge, x_limits, y_limits; panel_label)
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

function add_supercritical_markers!(axis, eigenvalues, targets)
    scatter!(
        axis,
        targets,
        zeros(length(targets));
        color=:red,
        marker=:star5,
        markersize=18.0,
        strokewidth=0,
    )

    matched = [eigenvalues[argmin(abs.(eigenvalues .- target))] for target in targets]
    scatter!(
        axis,
        real.(matched),
        imag.(matched);
        color=:blue,
        markersize=9.0,
        strokewidth=0,
    )
end

function generate_wigner_figure1(; n=1000, output_paths=DEFAULT_WIGNER_OUTPUTS)
    n >= 2 || error("n must be at least 2")
    iseven(n) || error("n must be even")
    length(output_paths) == 2 || error("output_paths must contain two paths")

    variance_profiles = [
        make_block_variance_profile(n, n, kappa) for kappa in KAPPAS
    ]

    eigenvalues = Matrix{Vector{ComplexF64}}(undef, 2, 3)
    targets = Matrix{Vector{Float64}}(undef, 2, 3)
    edges = zeros(2, 3)
    for row in 1:2
        T = variance_profiles[row]
        @assert issymmetric(T)
        @assert minimum(T) > 0.0
        T_norm = variance_profile_opnorm(T)

        for panel in 1:3
            eigenvalues[row, panel], targets[row, panel], edges[row, panel] =
                wigner_single_spike_realization(
                    n,
                    T,
                    T_norm,
                    GAMMAS[panel],
                    PANEL_SEEDS[panel],
                )
        end
    end

    x_values = Float64[]
    y_values = Float64[]
    for row in 1:2, panel in 1:3
        append!(x_values, real.(eigenvalues[row, panel]))
        append!(x_values, [-edges[row, panel], edges[row, panel]])
        append!(y_values, imag.(eigenvalues[row, panel]))
        append!(y_values, [-edges[row, panel], edges[row, panel]])
        if GAMMAS[panel] > 1.0
            append!(x_values, targets[row, panel])
        end
    end
    x_limits = padded_limits(x_values)
    y_limits = padded_limits(y_values)

    gamma = "\u03b3"
    panel_titles = ["$(gamma) = $(@sprintf("%.2f", value))" for value in GAMMAS]
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
            fig = Figure(size=(1500, 450), fontsize=22, figure_padding=20)
            for panel in 1:3
                axis = Axis(
                    fig[1, panel];
                    title=panel_titles[panel],
                    titlefont=:regular,
                    titlesize=25,
                    titlegap=10,
                    xlabel="Re",
                    ylabel="Im",
                    xlabelsize=24,
                    ylabelsize=24,
                    xticklabelsize=19,
                    yticklabelsize=19,
                    xgridcolor=(:gray, 0.18),
                    ygridcolor=(:gray, 0.18),
                    aspect=DataAspect(),
                )
                hidespines!(axis, :t, :r)
                draw_panel!(
                    axis,
                    eigenvalues[row, panel],
                    edges[row, panel],
                    x_limits,
                    y_limits;
                    panel_label=string('A' + 3 * (row - 1) + panel - 1),
                )
                if GAMMAS[panel] > 1.0
                    add_supercritical_markers!(
                        axis,
                        eigenvalues[row, panel],
                        targets[row, panel],
                    )
                end
            end
            colgap!(fig.layout, 24)
            for panel in 1:3
                colsize!(fig.layout, panel, Relative(1.0 / 3.0))
            end
            fig
        end for row in 1:2]
    end

    for (output_path, figure) in zip(output_paths, figures)
        mkpath(dirname(output_path))
        save(output_path, figure)
        println("Wigner phase-transition figure written to $(abspath(output_path))")
    end
    return output_paths
end

function generate_rectangular_figure1(; n=1000, output_paths=DEFAULT_OUTPUTS)
    n >= 10 || error("n must be at least 10")
    n % 10 == 0 || error("n must be divisible by 10")
    length(output_paths) == 2 || error("output_paths must contain two paths")
    p = 3 * n ÷ 5

    variance_profiles = [
        make_block_variance_profile(p, n, kappa) for kappa in KAPPAS
    ]

    eigenvalues = Matrix{Vector{ComplexF64}}(undef, 2, 3)
    targets = Matrix{Vector{Float64}}(undef, 2, 3)
    edges = zeros(2, 3)
    for row in 1:2
        T = variance_profiles[row]
        @assert size(T) == (p, n)
        @assert minimum(T) > 0.0
        T_norm = variance_profile_opnorm(T)

        for panel in 1:3
            eigenvalues[row, panel], targets[row, panel], edges[row, panel] =
                rectangular_single_spike_realization(
                    T,
                    T_norm,
                    GAMMAS[panel],
                    PANEL_SEEDS[panel],
                )
        end
    end

    x_values = Float64[]
    y_values = Float64[]
    for row in 1:2, panel in 1:3
        append!(x_values, real.(eigenvalues[row, panel]))
        append!(x_values, [-edges[row, panel], edges[row, panel]])
        append!(y_values, imag.(eigenvalues[row, panel]))
        append!(y_values, [-edges[row, panel], edges[row, panel]])
        if GAMMAS[panel] > 1.0
            append!(x_values, targets[row, panel])
        end
    end
    x_limits = padded_limits(x_values)
    y_limits = padded_limits(y_values)

    gamma = "\u03b3"
    panel_titles = ["$(gamma) = $(@sprintf("%.2f", value))" for value in GAMMAS]
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
            fig = Figure(size=(1500, 450), fontsize=22, figure_padding=20)
            for panel in 1:3
                axis = Axis(
                    fig[1, panel];
                    title=panel_titles[panel],
                    titlefont=:regular,
                    titlesize=25,
                    titlegap=10,
                    xlabel="Re",
                    ylabel="Im",
                    xlabelsize=24,
                    ylabelsize=24,
                    xticklabelsize=19,
                    yticklabelsize=19,
                    xgridcolor=(:gray, 0.18),
                    ygridcolor=(:gray, 0.18),
                    aspect=DataAspect(),
                )
                hidespines!(axis, :t, :r)
                draw_panel!(
                    axis,
                    eigenvalues[row, panel],
                    edges[row, panel],
                    x_limits,
                    y_limits;
                    panel_label=string('A' + 3 * (row - 1) + panel - 1),
                )
                if GAMMAS[panel] > 1.0
                    add_supercritical_markers!(
                        axis,
                        eigenvalues[row, panel],
                        targets[row, panel],
                    )
                end
            end
            colgap!(fig.layout, 24)
            for panel in 1:3
                colsize!(fig.layout, panel, Relative(1.0 / 3.0))
            end
            fig
        end for row in 1:2]
    end

    for (output_path, figure) in zip(output_paths, figures)
        mkpath(dirname(output_path))
        save(output_path, figure)
        println("Phase-transition figure written to $(abspath(output_path))")
    end
    return output_paths
end

function generate_figure1(
    ;
    n=1000,
    wigner_output_paths=DEFAULT_WIGNER_OUTPUTS,
    output_paths=DEFAULT_OUTPUTS,
)
    n >= 10 || error("n must be at least 10")
    n % 10 == 0 || error("n must be divisible by 10")
    wigner_outputs = generate_wigner_figure1(; n=n, output_paths=wigner_output_paths)
    rectangular_outputs = generate_rectangular_figure1(; n=n, output_paths=output_paths)
    return (wigner=wigner_outputs, rectangular=rectangular_outputs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_figure1()
end
