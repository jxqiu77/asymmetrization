#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    "Algorithm1.jl"
    "Algorithm2.jl"
    "Algorithm3.jl"
    "Algorithm4.jl"
    "Algorithm1_Comparison.jl"
    "Algorithm3_Comparison.jl"
    "phase_transition.jl"
    "distinct_repeated_spike.jl"
)

for script in "${scripts[@]}"; do
    printf 'Running %s\n' "$script"
    julia --startup-file=no --project="$project_dir" "$project_dir/code/$script"
done

printf 'Drawing comparison figures\n'
Rscript --vanilla "$project_dir/code/violin_plot.R"
printf 'Finished. Results: %s/output/csv; figures: %s/output/figure\n' "$project_dir" "$project_dir"
