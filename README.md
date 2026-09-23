# Spike Estimation from Heteroscedastic Noise via Random Splitting

Code for the paper "Spike Estimation from Heteroscedastic Noise via Random Splitting" ([arXiv:2609.11169](https://arxiv.org/abs/2609.11169)) by Zhigang Bao, Kha Man Cheong, Yuji Li, and Jiaxin Qiu.
The simulation results and figures are included.

## 📁 Project Structure

```text
.
|-- code/
|   |-- Algorithm1.jl               # Wigner spike estimation
|   |-- Algorithm2.jl               # Wigner overlaps and signal correlation
|   |-- Algorithm3.jl               # Rectangular spike estimation
|   |-- Algorithm4.jl               # Rectangular overlaps and signal correlation
|   |-- Algorithm1_Comparison.jl    # Comparison with BGS25 and BGN11
|   |-- Algorithm3_Comparison.jl    # Comparison with SN13 and OptShrink
|   |-- asymmetrization_utils.jl    # Shared functions
|   |-- phase_transition.jl         # Single-spike spectral plots
|   |-- distinct_repeated_spike.jl  # Distinct and repeated spike plots
|   `-- violin_plot.R               # Comparison plots from saved CSVs
|-- output/
|   |-- csv/                        # Six result CSVs
|   `-- figure/                     # Ten figure PDFs
|-- Project.toml                    # Julia dependencies
|-- Manifest.toml                   # Julia package versions
`-- run_all.sh                      # Run all simulations and plots
```

## 🚀 Quick Start

Run the commands below from the repository root.
Tested with Julia 1.12.7 and R 4.4.2.

```sh
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
Rscript -e 'install.packages(c("ggplot2", "showtext", "curl", "jsonlite"), repos="https://cloud.r-project.org")'

# Run all simulations and generate all figures
bash run_all.sh
```

Results are saved to `output/csv/` and `output/figure/`.

To run a single experiment, for example:

```sh
julia --project=. code/Algorithm1.jl
```

To regenerate only the figures:

```sh
julia --project=. code/phase_transition.jl
julia --project=. code/distinct_repeated_spike.jl
Rscript code/violin_plot.R
```

Contact: Jiaxin Qiu, <jxqiu77@gmail.com>.
