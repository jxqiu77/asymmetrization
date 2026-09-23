#!/usr/bin/env Rscript

# Plot the replication-level Algorithm 1 and Algorithm 3 comparisons at n = 1000.
#
# Usage:
#   Rscript code/violin_plot.R [input_csv] [output_dir]
#
# Outputs:
#   output/figure/algorithm1_comparison_violin_boxplot.pdf
#   output/figure/algorithm3_comparison_violin_boxplot.pdf

### load font
library(showtext)
## Loading Google fonts (https://fonts.google.com/)
font_add_google("Josefin Sans", "josefin")

### self-defined ggplot theme
library(ggplot2)
my_ggplot_theme <- theme_bw(
  base_size = 15,
  base_family = "josefin"
) +
  theme(
    plot.title = element_text(size = 20),
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 15),
    strip.text = element_text(size = 17),
    strip.placement = "outside",
    panel.grid.minor = element_blank(),
    panel.spacing.x = grid::unit(1.2, "lines"),
    panel.spacing.y = grid::unit(1.2, "lines"),
    legend.text = element_text(size = 15),
    legend.title = element_text(size = 20),
    legend.box.background = element_rect(),
    plot.margin = margin(12, 14, 12, 12)
  )

## Automatically use showtext to render text for future devices
showtext_auto()

comparison_kappas <- c(1L, 4L, 8L)
comparison_laws <- c("gaussian", "rademacher")
algorithm1_methods <- c("Algorithm 1", "BGS25", "BGN11")
algorithm3_methods <- c(
  "Algorithm 3",
  "SN13",
  "OptShrink"
)

# Okabe--Ito palette: distinguishable under common forms of color blindness.
algorithm1_colors <- c(
  "Algorithm 1" = "#0072B2",
  "BGS25" = "#D55E00",
  "BGN11" = "#009E73"
)
algorithm3_colors <- c(
  "Algorithm 3" = "#0072B2",
  "SN13" = "#D55E00",
  "OptShrink" = "#009E73"
)
profile_labels <- c(
  "1" = "T^{(1)}",
  "4" = "T^{(4)}",
  "8" = "T^{(8)}"
)
law_labels <- c(
  "gaussian" = "Gaussian",
  "rademacher" = "Rademacher"
)

script_path <- function() {
  file_argument <- grep(
    "^--file=",
    commandArgs(trailingOnly = FALSE),
    value = TRUE
  )
  if (length(file_argument) != 1L) {
    stop("Run this file with Rscript so its default paths can be resolved.")
  }
  normalizePath(sub("^--file=", "", file_argument[[1L]]), mustWork = TRUE)
}

read_comparison_estimates <- function(
  path,
  methods,
  expected_replications = 500L,
  expected_p = NULL
) {
  if (!file.exists(path)) {
    stop(sprintf("Comparison CSV not found: %s", normalizePath(path, mustWork = FALSE)))
  }

  records <- read.csv(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  required_columns <- c(
    "n",
    "kappa",
    "law",
    "strength",
    "replication",
    "method",
    "estimate"
  )
  if (!is.null(expected_p)) {
    required_columns <- c(required_columns, "p")
  }
  missing_columns <- setdiff(required_columns, names(records))
  if (length(missing_columns) > 0L) {
    stop(sprintf(
      "Comparison CSV is missing required columns: %s",
      paste(missing_columns, collapse = ", ")
    ))
  }

  numeric_columns <- c("n", "kappa", "strength", "replication", "estimate")
  if (!is.null(expected_p)) {
    numeric_columns <- c(numeric_columns, "p")
  }
  if (!all(vapply(records[numeric_columns], is.numeric, logical(1)))) {
    stop("The numeric CSV columns were not parsed as numbers.")
  }
  if (nrow(records) == 0L) {
    stop("The comparison CSV contains no data rows.")
  }
  if (!all(records$n == 1000)) {
    stop("The plotting data must contain only n = 1000.")
  }
  if (!setequal(unique(records$kappa), comparison_kappas)) {
    stop("Unexpected variance-profile grid.")
  }
  if (!setequal(unique(records$law), comparison_laws)) {
    stop("Unexpected noise-law grid.")
  }
  if (!setequal(unique(records$method), methods)) {
    stop("Unexpected comparison methods.")
  }
  if (!is.null(expected_p) && !all(records$p == expected_p)) {
    stop(sprintf("The plotting data must contain only p = %d.", expected_p))
  }
  if (
    any(!is.finite(records$strength)) ||
      any(!is.finite(records$estimate)) ||
      any(records$strength <= 0)
  ) {
    stop("The plotting data contain invalid strength or estimate values.")
  }

  replication_key <- paste(
    records$kappa,
    records$law,
    records$method,
    records$replication,
    sep = "\r"
  )
  if (anyDuplicated(replication_key)) {
    stop("Duplicate replication rows were found.")
  }

  group_key <- interaction(
    records$kappa,
    records$law,
    records$method,
    drop = TRUE,
    lex.order = TRUE
  )
  group_counts <- as.integer(table(group_key))
  expected_groups <-
    length(comparison_kappas) *
    length(comparison_laws) *
    length(methods)
  if (length(group_counts) != expected_groups) {
    stop("One or more profile-law-method configurations are missing.")
  }
  if (length(unique(group_counts)) != 1L) {
    stop("Replication counts are unbalanced across configurations.")
  }
  replication_count <- unique(group_counts)
  if (replication_count != expected_replications) {
    stop(sprintf(
      "Expected %d replications per configuration, found %d.",
      expected_replications,
      replication_count
    ))
  }
  expected_rows <- replication_count * expected_groups
  if (nrow(records) != expected_rows) {
    stop(sprintf("Expected %d rows, found %d.", expected_rows, nrow(records)))
  }

  records$normalized_estimate <- records$estimate / records$strength
  records$method_label <- factor(
    records$method,
    levels = methods
  )
  records$profile_label <- factor(
    profile_labels[as.character(records$kappa)],
    levels = unname(profile_labels)
  )
  records$law_label <- factor(
    law_labels[records$law],
    levels = unname(law_labels)
  )

  attr(records, "replication_count") <- replication_count
  attr(records, "configuration_count") <- expected_groups
  records
}

normalized_estimate_limits <- function(...) {
  records <- list(...)
  ratio_range <- range(unlist(lapply(
    records,
    function(data) data$normalized_estimate
  )))
  y_step <- 0.05
  c(
    floor(ratio_range[[1L]] / y_step) * y_step,
    ceiling(ratio_range[[2L]] / y_step) * y_step
  )
}

make_normalized_estimate_plot <- function(
  records,
  colors,
  y_limits,
  x_text_size = 15
) {
  y_step <- 0.05
  panel_labels <- unique(records[c("profile_label", "law_label")])
  panel_labels <- panel_labels[
    order(panel_labels$law_label, panel_labels$profile_label),
  ]
  panel_labels$label <- LETTERS[seq_len(nrow(panel_labels))]
  ggplot(
    records,
    aes(
      x = method_label,
      y = normalized_estimate,
      color = method_label,
      fill = method_label
    )
  ) +
    geom_hline(
      yintercept = 1,
      color = "black",
      linetype = "dashed",
      linewidth = 0.6
    ) +
    geom_violin(
      width = 0.8,
      scale = "width",
      trim = TRUE,
      alpha = 0.32,
      linewidth = 0.55
    ) +
    geom_boxplot(
      width = 0.16,
      outlier.shape = NA,
      fill = "white",
      linewidth = 0.55
    ) +
    geom_text(
      data = panel_labels,
      aes(label = label),
      x = -Inf,
      y = Inf,
      hjust = -0.35,
      vjust = 1.2,
      family = "josefin",
      fontface = "bold",
      size = 7,
      color = "black",
      inherit.aes = FALSE
    ) +
    facet_grid(
      law_label ~ profile_label,
      switch = "y",
      drop = FALSE,
      labeller = labeller(profile_label = label_parsed)
    ) +
    scale_color_manual(values = colors, drop = FALSE) +
    scale_fill_manual(values = colors, drop = FALSE) +
    scale_x_discrete(drop = FALSE) +
    scale_y_continuous(
      breaks = seq(y_limits[[1L]], y_limits[[2L]], by = y_step),
      expand = expansion(mult = 0)
    ) +
    coord_cartesian(ylim = y_limits, expand = FALSE) +
    labs(x = NULL, y = "Normalized estimate") +
    guides(color = "none", fill = "none") +
    my_ggplot_theme +
    theme(
      axis.text.x = element_text(size = x_text_size),
      panel.grid.major.x = element_blank()
    )
}

save_pdf <- function(plot, output_dir, stem) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  pdf_path <- file.path(output_dir, paste0(stem, ".pdf"))

  ggsave(
    pdf_path,
    plot,
    device = grDevices::pdf,
    width = 13.5,
    height = 8,
    units = "in",
    bg = "white"
  )
  message(sprintf("Figure written to %s", normalizePath(pdf_path)))

  invisible(pdf_path)
}

main <- function() {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) > 2L) {
    stop(
      "Usage: Rscript code/violin_plot.R [input_csv] [output_dir]"
    )
  }

  script_directory <- dirname(script_path())
  input_path <- if (length(arguments) >= 1L) {
    arguments[[1L]]
  } else {
    file.path(dirname(script_directory), "output", "csv", "algorithm1_comparison.csv")
  }
  output_dir <- if (length(arguments) >= 2L) {
    arguments[[2L]]
  } else {
    file.path(dirname(script_directory), "output", "figure")
  }

  algorithm3_input_path <- file.path(
    dirname(input_path),
    "algorithm3_comparison.csv"
  )
  algorithm1_records <- read_comparison_estimates(
    input_path,
    algorithm1_methods
  )
  algorithm3_records <- read_comparison_estimates(
    algorithm3_input_path,
    algorithm3_methods,
    expected_p = 600L
  )
  shared_y_limits <- normalized_estimate_limits(
    algorithm1_records,
    algorithm3_records
  )

  message(sprintf(
    "Validated Algorithm 1: %d rows, %d replications in each of %d configurations.",
    nrow(algorithm1_records),
    attr(algorithm1_records, "replication_count"),
    attr(algorithm1_records, "configuration_count")
  ))
  message(sprintf(
    "Validated Algorithm 3: %d rows, %d replications in each of %d configurations.",
    nrow(algorithm3_records),
    attr(algorithm3_records, "replication_count"),
    attr(algorithm3_records, "configuration_count")
  ))
  message(sprintf(
    "Shared normalized-estimate range: [%.2f, %.2f].",
    shared_y_limits[[1L]],
    shared_y_limits[[2L]]
  ))

  save_pdf(
    make_normalized_estimate_plot(
      algorithm1_records,
      algorithm1_colors,
      shared_y_limits
    ),
    output_dir,
    "algorithm1_comparison_violin_boxplot"
  )
  save_pdf(
    make_normalized_estimate_plot(
      algorithm3_records,
      algorithm3_colors,
      shared_y_limits,
      x_text_size = 13
    ),
    output_dir,
    "algorithm3_comparison_violin_boxplot"
  )
  invisible(NULL)
}

main()
