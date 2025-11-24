# =============================================================================
#
# Helper Functions for Sequential Forecast Calibration Analysis
#
# Description:
# This script provides a collection of functions to process raw forecast outputs,
# compute calibration diagnostics (PITs and ranks), and calculate sequential
# e-values for testing forecast calibration.
#
# =============================================================================


# ---- Data Processing and PIT/Rank Calculation -------------------------------

#' Calculate PIT for a Gaussian Forecast
#'
#' Calculates the Probability Integral Transform (PIT) for an observation
#' assuming the predictive distribution is Gaussian.
#'
#' @param y_true The observed outcome value.
#' @param prediction The predicted mean of the Gaussian distribution.
#' @param h_step_sd The predicted standard deviation of the Gaussian distribution.
#' @return A numeric vector of PIT values, ranging from 0 to 1.
gaussian_to_pit <- function(y_true, prediction, h_step_sd) {
  pnorm(y_true, mean = prediction, sd = h_step_sd)
}

#' Parse Numeric Draws from a String
#'
#' Converts a string representation of a numeric array (e.g., "[1, 2, 3]")
#' into a numeric matrix. This is used for processing the non-parametric PNC
#' model forecasts, which are stored as strings.
#'
#' @param draw_string A character vector where each element is a string array.
#' @return A numeric matrix where each row corresponds to an original string.
parse_draws <- function(draw_string) {
  # Strip brackets, split on commas, coerce numeric
  clean  <- gsub("\\[|\\]", "", draw_string)
  pieces <- strsplit(clean, ",\\s*")
  # rbind rows
  do.call(rbind, lapply(pieces, as.numeric))
}


#' Calculate Randomized Ranks for Ensemble Forecasts
#'
#' Computes the rank of the true value within a set of ensemble forecast samples.
#' This is a key diagnostic for non-parametric forecasts (e.g., PNC). The function
#' implements randomized tie-breaking to ensure the resulting ranks are uniformly
#' distributed under the null hypothesis of calibration.
#'
#' @param y_true A numeric vector of the true observed outcomes.
#' @param samples_matrix A numeric matrix of ensemble draws, where each row
#'   corresponds to an observation in `y_true`.
#' @return A numeric vector of randomized ranks.
pnc_to_ranks <- function(y_true, samples_matrix) {
  # Vectorized calculation of counts below and ties for all rows at once
  below <- rowSums(samples_matrix < y_true)
  ties <- rowSums(samples_matrix == y_true)
  
  # For rows with ties, generate a random integer from 0 to 'ties'
  random_tie_break <- floor(runif(length(y_true)) * (ties + 1))
  
  # The rank is the number of samples below + the random tie break + 1
  r <- below + random_tie_break + 1
  return(r)
}

#' Compute a Randomized PIT from a Discrete Predictive Distribution
#'
#' Calculates a randomized PIT value from a discrete distribution, typically
#' represented by a set of weighted outcomes (e.g., from a DRF model).
#' This correctly implements the PIT for discontinuous CDFs
#'
#' @param y_true The true observed outcome.
#' @param dist_list A list where each element contains the predictive distribution
#'   (a data frame with outcomes `y` and weights `w`) for the corresponding `y_true`.
#' @return A numeric vector of randomized PIT values.
compute_pit_from_distlist <- function(y_true, dist_list) {
  # Pre-generate random numbers for randomization component
  V <- runif(length(y_true))
  out <- numeric(length(y_true))
  
  for (i in seq_along(y_true)) {
    dist <- dist_list[[i]]
    y_pred <- dist$y
    w_pred <- dist$w
    
    # --- Input Validation ---
    if (!is.numeric(y_pred) || !is.numeric(w_pred) || length(y_pred) != length(w_pred) || length(y_pred) == 0) {
      out[i] <- NA_real_
      next
    }
    s <- sum(w_pred)
    if (!is.finite(s) || s <= 0) {
      out[i] <- NA_real_
      next
    }
    # Normalize weights to ensure they sum to 1
    w_pred <- w_pred / s
    
    # --- PIT Calculation ---
    f_y_minus <- sum(w_pred[y_pred < y_true[i]])
    f_y_prob_mass <- sum(w_pred[y_pred == y_true[i]])
    
    out[i] <- f_y_minus + V[i] * f_y_prob_mass
  }
  return(out)
}

# ---- Data Preparation Wrappers ----------------------------------------------

#' Prepare Data from Gaussian Forecasts
#'
#' This function acts as a data validation and preparation pipeline for forecasts
#' that assume a Gaussian predictive distribution. It filters data by horizon
#' and date, checks for required columns, removes non-finite values, and
#' computes PIT values.
#'
#' @param df The input data frame.
#' @param horizon The forecast horizon to filter for.
#' @param start_date The start date for the evaluation window.
#' @param end_date The end date for the evaluation window.
#' @return A list containing the vector of PIT values and corresponding target dates.
prepare_gaussian_data <- function(df, horizon, start_date, end_date) {
  if ("horizon_step" %in% names(df)) {
    df <- df[df$horizon_step == horizon, , drop = FALSE]
  }
  
  required_cols <- c("prediction", "h_step_sd", "y_true", "target_time")
  if (!all(required_cols %in% names(df))) return(NULL)
  
  df <- df[is.finite(df$prediction) & is.finite(df$h_step_sd) & is.finite(df$y_true), ]
  if (nrow(df) == 0) return(NULL)
  
  df$target_time <- as.Date(df$target_time)
  if (!is.null(start_date)) df <- df[df$target_time >= as.Date(start_date), ]
  if (!is.null(end_date)) df <- df[df$target_time <= as.Date(end_date), ]
  if (nrow(df) == 0) return(NULL)
  
  list(
    pit_values = gaussian_to_pit(df$y_true, df$prediction, df$h_step_sd),
    target_dates = df$target_time
  )
}

#' Prepare Data from PNC (Ensemble) Forecasts
#'
#' Validation and preparation pipeline for the Probabilistic No-Change (PNC)
#' model. It filters data, validates inputs, parses the ensemble draws from
#' strings, and computes randomized ranks.
#'
#' @param df The input data frame.
#' @param horizon The forecast horizon.
#' @param start_date The start date for the evaluation window.
#' @param end_date The end date for the evaluation window.
#' @return A list containing ranks, dates, and the number of histogram bins `m`.
prepare_pnc_data <- function(df, horizon, start_date, end_date) {
  if ("horizon_step" %in% names(df)) {
    df <- df[df$horizon_step == horizon, , drop = FALSE]
  }
  
  required_cols <- c("last_20_values", "y_true", "target_time")
  if (!all(required_cols %in% names(df))) return(NULL)
  
  df <- df[is.finite(df$y_true) & !is.na(df$last_20_values) & nzchar(df$last_20_values), ]
  if (nrow(df) == 0) return(NULL)
  
  df$target_time <- as.Date(df$target_time)
  if (!is.null(start_date)) df <- df[df$target_time >= as.Date(start_date), ]
  if (!is.null(end_date)) df <- df[df$target_time <= as.Date(end_date), ]
  if (nrow(df) == 0) return(NULL)
  
  draws <- parse_draws(df$last_20_values)
  if (!is.matrix(draws)) draws <- matrix(draws, nrow = nrow(df), byrow = TRUE)
  
  list(
    ranks = pnc_to_ranks(df$y_true, draws),
    target_dates = df$target_time,
    m_bins = ncol(draws) + 1 # Rank is in {1, ..., m+1}
  )
}

#' Prepare Data from DRF Forecasts
#'
#' Validation and preparation pipeline for the Distributional Random Forest (DRF)
#' model. It aligns forecast metadata with the distributional objects, filters
#' by date, and computes randomized PIT values from the discrete predictive distributions.
#'
#' @param drf_results Data frame with forecast metadata (e.g., `y_true`, `target_time`).
#' @param drf_dists List of distributional objects corresponding to `drf_results`.
#' @param horizon The forecast horizon.
#' @param start_date The start date for the evaluation window.
#' @param end_date The end date for the evaluation window.
#' @return A list containing PIT values and corresponding target dates.
prepare_drf_data <- function(drf_results, drf_dists, horizon, start_date, end_date) {
  if ("horizon_step" %in% names(drf_results)) {
    drf_results <- drf_results[drf_results$horizon_step == horizon, , drop = FALSE]
  }
  
  if (nrow(drf_results) != length(drf_dists)) stop("DRF results and distributions have mismatched lengths.")
  
  drf_results$target_time <- as.Date(drf_results$target_time)
  if (!is.null(start_date)) {
    keep_idx <- drf_results$target_time >= as.Date(start_date)
    drf_results <- drf_results[keep_idx, ]
    drf_dists <- drf_dists[keep_idx]
  }
  if (!is.null(end_date)) {
    keep_idx <- drf_results$target_time <= as.Date(end_date)
    drf_results <- drf_results[keep_idx, ]
    drf_dists <- drf_dists[keep_idx]
  }
  if (nrow(drf_results) == 0) return(NULL)
  
  list(
    pit_values = compute_pit_from_distlist(drf_results$y_true, drf_dists),
    target_dates = drf_results$target_time
  )
}

# ---- Metadata & String Helpers --------------------------------------------
#' Categorize a Model into a Group
#'
#' Assigns a model to a broader group based on its name.
#' Useful for organizing and faceting plots.
#' @param name The model name string.
#' @return A character string for the model group.
detect_group <- function(name) {
  n <- tolower(name)
  if (grepl("bvar", n)) "BVAR"
  else if (grepl("auto[_-]?arima", n) || grepl("naive.*mean|naive.*last|no[_-]?change|pnc", n)) "Baseline"
  else if (grepl("arima[_-]?110|arima\\s*1\\s*1\\s*0", n) || grepl("drf", n) || grepl("dfm", n)) "Advanced"
  else "Other"
}

#' Create a Publication-Ready Model Name
#'
#' Converts a technical model filename into a clean, formatted name suitable for plots and tables.
#' @param name The model name string.
#' @return A formatted character string.
pretty_model <- function(name) {
  n <- tolower(name)
  if (grepl("bvar.*diffuse", n)) "BVAR (diffuse)"
  else if (grepl("bvar.*minnesota", n)) "BVAR (Minnesota)"
  else if (grepl("bvar.*normalwishart|bvar.*normal_wishart", n)) "BVAR (NW)"
  else if (grepl("auto[_-]?arima", n)) "Auto-ARIMA"
  else if (grepl("arima[_-]?110|arima\\s*1\\s*1\\s*0", n)) "ARIMA(1,1,0)"
  else if (grepl("naive.*mean", n)) "Rolling mean"
  else if (grepl("naive.*last|no[_-]?change", n)) "Naive last"
  else if (grepl("pnc", n)) "PNC"
  else if (grepl("drf", n)) "DRF"
  else if (grepl("dfm", n)) "DFM"
  else basename(name)
}

# ---- Core E-Value Calculation and Aggregation -------------------------------

#' Merge Sequential E-values into a Cumulative E-Process
#'
#' This function correctly aggregates sequential e-values into a single cumulative
#' e-process. For single-step forecasts (`h=1`), it computes the cumulative product.
#' For multi-step forecasts (`h>1`), it uses the U-statistic-based aggregation method
#' as implemented in the `epit` package.
#'
#' @param e_output An object from an `e_*` function in the `epit` package.
#' @return A numeric vector representing the cumulative e-process over time.
e_vec_merge <- function(e_output) {
  # For h=1, the e-process is a test supermartingale formed by the cumulative product.
  if (e_output$h == 1) {
    return(base::cumprod(e_output$e))
  }
  # For h>1, use the implementation for merging lagged e-value sequences.
  es <- lapply(e_output$evalues_h, function(x) x$e)
  epit:::evalue_combine_h(es)
}

#' Master Function to Compute Calibration Series
#'
#' This is a high-level wrapper that orchestrates the entire calibration analysis
#' for a given model. It automatically detects the forecast type, dispatches to the
#' appropriate data preparation function, computes the corresponding sequential
#' e-values, and returns a structured list containing all results.
#'
#' @param name The name of the model being analyzed.
#' @param df The data frame of forecast results.
#' @param type The type of forecast ('auto', 'gaussian', 'pnc', 'drf').
#' @param horizon The forecast horizon (integer).
#' @param drf_results For DRF models, the data frame of results.
#' @param drf_dists For DRF models, the list of distribution objects.
#' @param n0 The warm-up period (number of initial observations) before e-value
#'   calculation begins. E-values during this period are set to 1.
#' @param start_date The start date of the analysis window.
#' @param end_date The end date of the analysis window.
#' @return A list containing `calibration_data` (PITs/ranks), `e` (the e-process),
#'   and `boundary_stats` (statistics on PITs at 0 or 1).
compute_model_series <- function(name, df = NULL, type = c("auto","gaussian","pnc","drf"),
                                 horizon = 1L, drf_results = NULL, drf_dists = NULL,
                                 n0 = 10L, start_date = NULL, end_date = NULL) {
  type <- match.arg(type)
  
  # --- Auto-detect model type if not specified ---
  if (type == "auto") {
    is_gauss <- !is.null(df) && all(c("prediction", "h_step_sd", "y_true") %in% names(df))
    is_pnc <- !is.null(df) && all(c("last_20_values", "y_true") %in% names(df))
    
    if (is_gauss) type <- "gaussian"
    else if (is_pnc) type <- "pnc"
    else stop("Cannot auto-detect model type for: ", name)
  }
  
  # --- Dispatch to the correct data preparation helper ---
  prepared_data <- switch(type,
                          gaussian = prepare_gaussian_data(df, horizon, start_date, end_date),
                          pnc = prepare_pnc_data(df, horizon, start_date, end_date),
                          drf = prepare_drf_data(drf_results, drf_dists, horizon, start_date, end_date),
                          stop("Unknown model type: ", type)
  )
  
  if (is.null(prepared_data)) return(NULL)
  
  # --- E-Value Calculation for Rank-based (PNC) Forecasts ---
  if (type == "pnc") {
    if (length(prepared_data$ranks) < 2) return(NULL)
    
    # Calculate e-values using the beta-binomial strategy for discrete/rank data
    bb_obj <- epit::e_rank_histogram(
      r = prepared_data$ranks, m = prepared_data$m_bins, h = horizon,
      strategy = "betabinom", options = list(n0 = n0)
    )
    
    rank_tibble <- tibble(
      model = pretty_model(name), group = detect_group(name),
      target_time = prepared_data$target_dates, value = prepared_data$ranks,
      m_bins = prepared_data$m_bins, type = "rank",
      horizon = horizon
    )
    
    # Calculate statistics on boundary ranks (1 or m)
    total_obs <- length(prepared_data$ranks)
    m <- prepared_data$m_bins
    boundary_count <- sum(prepared_data$ranks %in% c(1, m), na.rm = TRUE)
    boundary_stats <- tibble(
      boundary_count = boundary_count,
      total_observations = total_obs,
      boundary_share = if (total_obs > 0) boundary_count / total_obs else NA_real_
    )
    
    # Merge the sequential e-values into a final e-process
    e_tibble <- tibble(
      method = "rank_betabinom", target_time = prepared_data$target_dates,
      e = e_vec_merge(bb_obj), horizon = horizon
    ) %>% mutate(model = pretty_model(name), group = detect_group(name))
    
    return(list(
      calibration_data = rank_tibble, e = e_tibble, 
      boundary_stats = boundary_stats
    ))
    
  } else {
    # --- E-Value Calculation for PIT-based (Gaussian, DRF) Forecasts ---
    is_finite <- is.finite(prepared_data$pit_values)
    finite_pits <- prepared_data$pit_values[is_finite]
    finite_dates <- prepared_data$target_dates[is_finite]
    total_obs <- length(finite_pits)
    
    # Identify PIT values at the boundaries [0, 1]. These are excluded
    # from parameter estimation for the beta e-value to avoid divergence.
    is_boundary <- (finite_pits <= 0 | finite_pits >= 1)
    is_valid_for_epit <- !is_boundary
    
    boundary_count <- sum(is_boundary)
    boundary_share <- if (total_obs > 0) boundary_count / total_obs else 0
    boundary_stats <- tibble(
      boundary_count = boundary_count, total_observations = total_obs,
      boundary_share = boundary_share
    )
    
    pit_subset <- finite_pits[is_valid_for_epit]
    
    # Return early if not enough non-boundary data points for a meaningful test
    if (length(pit_subset) < n0) {
      cal_data <- if(length(pit_subset) > 0) {
        tibble(model=pretty_model(name), group=detect_group(name),
               target_time=finite_dates[is_valid_for_epit],
               value=pit_subset, type="pit")
      } else { tibble() }
      return(list(calibration_data = cal_data, e = tibble(),
                  boundary_stats = boundary_stats))
    }
    
    # Keep track of original indices to place calculated e-values back correctly
    pit_data_subset <- tibble(
      pit = pit_subset,
      original_index = which(is_valid_for_epit)
    )
    
    # Initialize a vector to hold the e-value increments for all observations
    e_values_full <- rep(1, total_obs)
    
    # This loop implements the procedure for multi-step forecasts (h>1).
    # It splits the data into `h` subsequences, and calculates an independent
    # stream of e-values for each, treating each as an h=1 problem.
    for (i in 1:horizon) {
      # Select the i-th subsequence
      sel <- which(((pit_data_subset$original_index - 1) %% horizon) == (i - 1))
      if (length(sel) >= n0) {
        z_stream <- pit_data_subset$pit[sel]
        
        # Calculate e-values for this subsequence using the beta strategy
        beta_obj_stream <- epit::e_pit(
          z = z_stream, h = 1, strategy = "beta", options = list(n0 = n0)
        )
        # Place the calculated e-values back into the full vector
        e_values_full[pit_data_subset$original_index[sel]] <- beta_obj_stream$e
      }
    }
    
    calibration_tibble <- tibble(
      model = pretty_model(name), group = detect_group(name),
      target_time = finite_dates[is_valid_for_epit], value = pit_subset,
      type = "pit", horizon = horizon
    )
    
    # --- Aggregate the e-value increments into the final e-process ---
    if (horizon == 1) {
      # For h=1, this is just the cumulative product of all e-value increments
      merged_e_process <- cumprod(e_values_full)
    } else {
      # For h>1, we must reconstruct the list of `h` separate e-value streams
      # to pass to the merging function, which performs the averaging of products.
      interleaved_e_list <- list()
      for (i in 1:horizon) {
        indices <- seq(from = i, to = total_obs, by = horizon)
        interleaved_e_list[[i]] <- list(e = e_values_full[indices]) 
      }
      e_merge_input <- list(h = horizon, evalues_h = interleaved_e_list)
      merged_e_process <- e_vec_merge(e_merge_input)
    }
    
    e_tibble <- tibble(
      method = "beta_e", target_time = finite_dates, e = merged_e_process,
      horizon = horizon
    ) %>% mutate(model = pretty_model(name), group = detect_group(name))
    
    return(list(
      calibration_data = calibration_tibble, e = e_tibble,
      boundary_stats = boundary_stats
    ))
  }
}

# ---- Statistical Test Helpers -----------------------------------------------

#' Perform Kolmogorov-Smirnov Test for Uniformity
#'
#' This function applies the one-sample Kolmogorov-Smirnov (KS) test to assess
#' whether a set of Probability Integral Transform (PIT) values deviates
#' significantly from the standard uniform distribution.
#'
#' @param pit_df A data frame containing at least two columns: `model` (character)
#'   and `value` (numeric PITs).
#' @return A tibble with columns for `model`, the `ks_statistic` (D), and the `p_value`.
calculate_ks_test <- function(pit_df) {
  if (nrow(pit_df) == 0) {
    return(tibble(model = character(), ks_statistic = numeric(), p_value = numeric()))
  }
  
  pit_df %>%
    dplyr::group_by(model) %>%
    dplyr::summarise({
      # Test the 'value' column against the theoretical CDF of a uniform distribution
      # exact=FALSE is used to avoid issues with ties
      test_result <- stats::ks.test(value, "punif", exact = FALSE)
      tibble(
        ks_statistic = test_result$statistic,
        p_value = test_result$p.value
      )
    }, .groups = "drop")
}

# ---- Plotting Helpers -----------------------------------------------------
#' Create Panel (a): PIT/Rank Calibration Histograms
#'
#' Generates the left-hand panel for calibration diagnostic plots, showing
#' PIT or rank histograms.
#'
#' @param calibration_df A data frame containing PITs and/or ranks from the
#'   `compute_model_series` function.
#' @param region_label An optional string (e.g., "CH", "EU", "US") to include in the plot title.
#' @param add_density A logical flag to control whether a boundary-corrected
#'   kernel density estimate is overlaid on the PIT histograms.
#' @return A ggplot object.
panel_a <- function(calibration_df, region_label = NULL, add_density = TRUE) {
  pit_data  <- calibration_df %>% dplyr::filter(type == "pit")
  rank_data <- calibration_df %>% dplyr::filter(type == "rank")
  
  lab_vals <- labeller(model = label_value, horizon = label_value)
  
  # Dynamically set x-axis label based on the content
  grp_vals <- unique(calibration_df$group)
  x_lab <- if (length(grp_vals) == 1 && grp_vals == "Baseline") {
    "PIT / Rank (scaled to [0,1])"
  } else {
    "PIT"
  }
  
  p <- ggplot() +
    facet_grid(model ~ horizon, scales = "free_y", labeller = lab_vals) +
    labs(
      x = x_lab, y = "Density",
      title = paste0("(a)", if (!is.null(region_label)) paste0(" ", toupper(region_label)) else "")
    ) +
    theme(
      strip.text.y    = element_text(angle = 270),
      panel.spacing.x = grid::unit(8, "pt")
    )
  
  # --- Layer 1: PIT Histogram and Density ---
  if (nrow(pit_data) > 0) {
    # Base layer: Histogram with uniform reference line
    p <- p +
      geom_histogram(
        data = pit_data, aes(x = value, y = after_stat(density)),
        binwidth = 0.05, boundary = 0
      ) +
      geom_hline(data = pit_data, aes(yintercept = 1), colour = "firebrick", linewidth = 0.8)
    
    # Optional Layer: Boundary-corrected kernel density overlay
    if (isTRUE(add_density)) {
      density_data <- pit_data %>%
        dplyr::group_by(model, horizon) %>%
        dplyr::reframe({
          tryCatch({
            z <- value[is.finite(value) & value > 0 & value < 1]
            if (length(z) < 5) return(tibble(x = numeric(), y = numeric()))
            grd <- seq(0.001, 1 - 0.001, length.out = 401)
            h <- KernSmooth::dpik(
              x = z, scalest = "min", level = 2L,
              kernel = "epanech", gridsize = 401L, range.x = c(0, 1)
            )
            if (h <= 0 || !is.finite(h)) return(tibble(x = numeric(), y = numeric()))
            # Use boundary-corrected kernel
            kernel_estim <- bde::jonesCorrectionMuller94BoundaryKernel(
              dataPoints = z, b = h, mu = 1, dataPointsCache = grd
            )
            tibble(x = grd, y = kernel_estim@densityCache)
          }, error = function(e) tibble(x = numeric(), y = numeric()))
        })
      
      pit_caps <- pit_data %>%
        dplyr::group_by(model, horizon) %>%
        dplyr::summarise(
          max_bin = {
            z <- value[is.finite(value) & value >= 0 & value <= 1]
            if (length(z) < 1) 0 else {
              hst <- hist(z,
                          breaks = seq(0, 1, 0.05),
                          include.lowest = TRUE, right = FALSE, plot = FALSE)
              max(hst$density, na.rm = TRUE)
            }
          },
          .groups = "drop"
        ) %>%
        dplyr::mutate(cap = 2 * max_bin)
      
      density_data <- density_data %>%
        dplyr::left_join(pit_caps, by = c("model", "horizon")) %>%
        dplyr::mutate(y_cap = ifelse(is.na(cap), y, pmin(y, cap)))
      
      p <- p + geom_line(data = density_data, aes(x = x, y = y_cap), linewidth = 0.8)
    }
  }
  
  # --- Layer 2: Rank Histogram (rescaled to [0,1] for comparability) ---
  if (nrow(rank_data) > 0) {
    mb <- unique(rank_data$m_bins); stopifnot(length(mb) == 1L)
    rank_scaled <- rank_data %>% mutate(value01 = (value - 0.5) / mb)
    p <- p +
      geom_histogram(
        data = rank_scaled, aes(x = value01, y = after_stat(density)),
        binwidth = 1 / mb, boundary = 0
      ) +
      geom_hline(yintercept = 1, colour = "firebrick", linewidth = 0.8)
  }
  
  p + scale_x_continuous(limits = c(0, 1), breaks = c(0.2, 0.5, 0.8))
}


#' Create Panel (b): E-Process Time Series Plot
#'
#' Generates the right-hand panel for calibration diagnostic plots, showing the
#' evolution of the cumulative e-process over time on a logarithmic scale.
#' This visualization reveals when and how evidence against the null hypothesis
#' of calibration accumulates.
#'
#' @param e_df A data frame of e-process values from `compute_model_series`.
#' @param region_label An optional string (e.g., "CH", "EU", "US") to help
#'   determine appropriate date axis breaks.
#' @return A ggplot object.
panel_b <- function(e_df, region_label = NULL) {
  # Filter out non-positive values which are invalid for a log scale
  e_df <- e_df |> dplyr::filter(is.finite(e), e > 0)
  lab_vals <- labeller(model = label_value, horizon = label_value)
  # Adapt date breaks to the time range of the data for better readability
  major_step <- if (!is.null(region_label) && tolower(region_label) == "us") "6 years" else "5 years"
  
  ggplot(e_df, aes(target_time, e)) +
    # Add reference lines at 1 (null hypothesis) and 100
    geom_hline(yintercept = c(1, 100), lty = 3, col = 1) +
    geom_line(linewidth = 0.8) +
    facet_grid(model ~ horizon, scales = "free_y", labeller = lab_vals) +
    scale_y_log10() +
    scale_x_date(
      breaks = scales::date_breaks(major_step),
      minor_breaks = scales::date_breaks("1 year"),
      labels = scales::label_date("’%y"),
      expand = expansion(mult = c(0.01, 0.02))
    ) +
    labs(x = "Date", y = "E-value", title = "(b)") +
    theme(
      strip.text.y    = element_text(angle = 270),
      legend.position = "none",
      panel.spacing.x = grid::unit(8, "pt")
    )
}

#' Create Panel (c): Time-Varying PIT Density Plot
#'
#' Replicates the Arnold et al. methodology for visualizing PIT density evolution.
#' It plots a non-parametric kernel density and a parametric Beta density fit.
#' The solid line shows the density for the specific period, and the dashed line
#' shows the density of all data prior to the start of that period.
#'
#' @param cal_data A data frame of PITs for a SINGLE model at a SINGLE horizon.
#' @param split_dates A vector of Date objects for time windows.
#' @param model_name The pretty name of the model for the plot title.
#' @param horizon The forecast horizon, included in the title.
#' @return A ggplot object.
panel_c_pit_evolution <- function(cal_data, split_dates, model_name, horizon,
                                  panel_index = 1, y_limit_max = 6) {
  
  # --- Helper function to calculate kernel density ---
  calculate_kernel_density <- function(data_subset) {
    tryCatch({
      z <- data_subset$value[is.finite(data_subset$value) & data_subset$value > 0 & data_subset$value < 1]
      if (length(z) < 10) return(tibble(x = numeric(), y = numeric()))
      
      grd <- seq(0.001, 0.999, length.out = 401)
      h <- KernSmooth::dpik(x = z, scalest = "min", range.x = c(0, 1))
      if (h <= 0 || !is.finite(h)) return(tibble(x = numeric(), y = numeric()))
      
      kernel_estim <- bde::jonesCorrectionMuller94BoundaryKernel(dataPoints = z, b = h, dataPointsCache = grd)
      tibble(x = grd, y = kernel_estim@densityCache)
    }, error = function(e) tibble(x = numeric(), y = numeric()))
  }
  
  # Helper function to calculate BETA density
  calculate_beta_density <- function(data_subset) {
    z <- data_subset$value[is.finite(data_subset$value) & data_subset$value > 0 & data_subset$value < 1]
    if (length(z) < 10) return(tibble(x = numeric(), y = numeric())) # Need enough data
    
    # Fit Beta distribution using MLE
    params <- tryCatch({
      fit <- MASS::fitdistr(z, "beta", start = list(shape1 = 1, shape2 = 1))
      as.list(fit$estimate)
    }, error = function(e) NULL)
    
    if (is.null(params)) return(tibble(x = numeric(), y = numeric()))
    
    grd <- seq(0.001, 0.999, length.out = 401)
    density_values <- dbeta(grd, shape1 = params$shape1, shape2 = params$shape2)
    
    tibble(x = grd, y = density_values)
  }
  
  # --- 1. Prepare time windows and data ---
  s_dates <- sort(as.Date(split_dates))
  breaks <- c(as.Date(-Inf), s_dates, as.Date(Inf))
  labels <- c(
    paste("<", format(s_dates[1], "%Y-%m")),
    paste(format(s_dates[-length(s_dates)], "%Y-%m"), format(s_dates[-1], "%Y-%m"), sep = " to "),
    paste(">", format(s_dates[length(s_dates)], "%Y-%m"))
  )
  
  base_data <- cal_data %>% filter(type == "pit")
  
  # --- 2. Loop through periods to calculate both period-specific and prior densities ---
  all_periods_density_data <- list()
  period_levels <- labels
  
  for (i in seq_along(period_levels)) {
    period_label <- period_levels[i]
    start_date_of_period <- breaks[i]
    end_date_of_period <- breaks[i + 1]
    
    # Data for the current and prior period
    period_data <- base_data %>% 
      filter(target_time > start_date_of_period & target_time <= end_date_of_period)
    prior_data <- base_data %>% filter(target_time <= start_date_of_period)
    
    # Calculate kernel and beta densities for both periods
    density_period_kern <- calculate_kernel_density(period_data)
    density_prior_kern  <- calculate_kernel_density(prior_data)
    density_period_beta <- calculate_beta_density(period_data)
    density_prior_beta  <- calculate_beta_density(prior_data)
    
    # Assign labels
    if (nrow(density_period_kern) > 0) {
      density_period_kern$type <- "Current period only"; density_period_kern$method <- "Kernel"
    }
    if (nrow(density_prior_kern) > 0) {
      density_prior_kern$type <- "Data from previous periods"; density_prior_kern$method <- "Kernel"
    }
    if (nrow(density_period_beta) > 0) {
      density_period_beta$type <- "Current period only"; density_period_beta$method <- "Beta"
    }
    if (nrow(density_prior_beta) > 0) {
      density_prior_beta$type <- "Data from previous periods"; density_prior_beta$method <- "Beta"
    }
    
    combined <- bind_rows(density_period_kern, density_prior_kern, density_period_beta, density_prior_beta)
    if (nrow(combined) > 0) {
      combined$time_period <- period_label
      all_periods_density_data[[i]] <- combined
    }
  }
  
  density_data <- bind_rows(all_periods_density_data)
  if (nrow(density_data) == 0) return(patchwork::plot_spacer())
  density_data$time_period <- factor(density_data$time_period, levels = labels)
  density_data$model_horizon_label <- sprintf("%s (h=%d)", model_name, horizon)
  
  # Find the actual highest density value in the data for this plot
  actual_max_y <- if (nrow(density_data) > 0) max(density_data$y, na.rm = TRUE) else 0
  
  # Create the dynamic limit for this plot.
  y_limit <- min(actual_max_y * 1.1, y_limit_max)
  
  peak_labels <- density_data %>%
    dplyr::filter(y > y_limit) %>%
    # Find the peak for EACH method 
    dplyr::group_by(time_period, model_horizon_label, type, method) %>%
    dplyr::summarise(
      peak_x = x[which.max(y)],
      true_max_y = max(y),
      .groups = "drop"
    ) %>%
    # Group again, but WITHOUT method, to find the max peak PER FACET
    dplyr::group_by(time_period, model_horizon_label, type) %>%
    #  Keep only the row with the highest true peak value
    dplyr::slice_max(order_by = true_max_y, n = 1, with_ties = FALSE) %>%
    dplyr::ungroup() %>%
    # Create nudged coordinates for the single remaining label 
    dplyr::mutate(
      y = y_limit * 0.9,
      x = dplyr::case_when(
        peak_x < 0.10 ~ 0.10,
        peak_x > 0.90 ~ 0.90,
        TRUE          ~ peak_x
      ),
      label = sprintf("%.1f", true_max_y)
    )
  
  density_data <- density_data %>%
    dplyr::mutate(y = pmin(y, y_limit))
  
  # dynamic title
  plot_title <- if (panel_index == 1) {
    sprintf("(c%d) PIT density over time", panel_index)
  } else {
    sprintf("(c%d)", panel_index)
  }
  
  # --- 3. Create the plot with final theme adjustments ---
  p <- ggplot(density_data, aes(x = x, y = y, linetype = type, color = method)) +
    geom_hline(yintercept = 1, colour = "firebrick", linewidth = 0.6) +
    geom_line(linewidth = 0.8) +
    geom_text(data = peak_labels, aes(label = label), size = 2.5, vjust = 0.5,
              hjust = 0.5) +
    facet_grid(model_horizon_label ~ time_period) +
    scale_linetype_manual(
      name = "Data:",
      values = c("Data from previous periods" = "dashed", "Current period only" = "solid")
    ) +
    scale_color_manual(
      name = "Method:",
      values = c("Kernel" = "black", "Beta" = "#0072B2"),
      labels = c("Beta" = "Beta (MLE)", "Kernel" = "Kernel (non-parametric)")
    ) +
    guides(
      linetype = guide_legend(title = "Data:", order = 1),
      color = guide_legend(title = "Method:", order = 2)
    ) +
    labs(
      x = "PIT", y = "Estimated density",
      # Title is dynamic
      title = plot_title
    ) +
    theme(
      panel.spacing.x = grid::unit(8, "pt"),
      legend.position = "bottom",
      legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
      plot.title = element_text(hjust = 0),
      strip.text.y    = element_text(angle = 270)
    ) +
    scale_x_continuous(limits = c(0, 1), breaks = c(0.2, 0.5, 0.8)) +
    scale_y_continuous(labels = scales::label_number(accuracy = 0.1))
  
  p <- p + coord_cartesian(ylim = c(0, y_limit))
  
  return(p)
}

# ---- Utility Helpers --------------------------------------------------------

#' Find the First Usable Forecast Date for a Gaussian Model
#'
#' Scans a forecast data frame to find the earliest date for which all necessary
#' components (prediction, sd, true value, date) are finite and non-missing.
#'
#' @param df A forecast data frame.
#' @param horizon The specific forecast horizon to check.
#' @return The earliest valid `Date` object, or `NA` if none are found.
usable_start_gaussian <- function(df, horizon) {
  if ("horizon_step" %in% names(df)) df <- df[df$horizon_step == horizon, , drop = FALSE]
  if (!all(c("prediction","h_step_sd","y_true","target_time") %in% names(df))) return(as.Date(NA))
  keep <- is.finite(df$prediction) & is.finite(df$h_step_sd) & is.finite(df$y_true) & !is.na(df$target_time)
  if (!any(keep)) return(as.Date(NA))
  min(as.Date(df$target_time[keep]))
}
