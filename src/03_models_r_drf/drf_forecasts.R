# ============================================================
# Inflation Forecasting - DRF Modeling Script (Direct Forecasting) - CORRECTED
# ============================================================
# This script loads prepared design matrices and performs direct, multi-horizon DRF modeling
# by fitting a separate model for each forecast horizon.
# Input: CSV files from data preparation script
# Output: DRF forecasts, variable importance, and visualizations

# ---- 0. Packages ----------------------------------------------------------
library(here)
library(readr)
library(dplyr)
library(tidyr)
library(drf)
library(ggplot2)
library(scoringRules)
library(lubridate)

# --- Load project-specific helpers ---
source("drf_helpers.R")

# ---- 1. Configuration -----------------------------------------------------
CONFIG <- list(
  # Paths
  prepared_data_dir = file.path(here::here("..", ".."), "data", "processed", "for_drf"),
  out_dir = file.path(here::here("..", ".."), "results", "forecasts", "drf"),
  
  # DRF parameters
  num_trees = 2000,
  splitting_rule = "FourierMMD",
  sample_fraction = 0.5,
  min_node_size = 5,
  
  # Forecast parameters
  quantiles = c(0.025, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.975),
  quantile_names = c("q025", "q05", "q10", "q20", "q25", "q30", "q40", "q50", "q60", "q70", "q75", "q80", "q90", "q95", "q975"),
  
  # Plotting
  vip_top_n = 25,
  
  # Other
  seed = 42
)


# ---- 2. Main Modeling Pipeline --------------------------------------------

#' Run DRF analysis on prepared data
run_drf_analysis <- function(regions = c("ch", "eu", "us"), save_results = TRUE) {
  if (!dir.exists(CONFIG$prepared_data_dir)) {
    stop(sprintf("Prepared data directory not found. Run data prep script first."))
  }
  
  design_data_list <- list()
  all_metrics <- list()
  
  for (region in regions) {
    tryCatch({
      design_data_list[[region]] <- load_design_matrix(region, CONFIG$prepared_data_dir)
    }, error = function(e) {
      cat(sprintf("Error loading data for %s: %s\n", toupper(region), e$message))
    })
  }
  if (length(design_data_list) == 0) stop("No valid data loaded.")
  
  full_forecast_output <- list()
  for (region in names(design_data_list)) {
    cat(sprintf("\n--- Processing Region: %s ---\n", toupper(region)))
    
    region_data <- design_data_list[[region]]
    region_results_list <- list()
    
    # List to collect results data frames
    region_results_df_list <- list()
    
    # --- MAIN HORIZON LOOP ---
    for (h in region_data$horizons) {
      cat(sprintf("\n-- Forecasting for horizon h=%d --\n", h))
      
      Y_h_full <- region_data$Y_df[[paste0("target_h", h)]]
      last_valid_idx <- max(which(!is.na(Y_h_full)))
      init_train_size <- region_data$metadata$init_train_size
      
      if (last_valid_idx <= init_train_size) {
        cat(sprintf("  Skipping h=%d: Not enough observations for a forecast.\n", h))
        next
      }
      
      X_h <- region_data$X[1:last_valid_idx, ]
      Y_h <- Y_h_full[1:last_valid_idx]
      dates_h <- region_data$dates[1:last_valid_idx]
      
      analyze_variable_importance(X_h, Y_h, region, h)
      
      forecast_output_h <- rolling_forecast(
        X_df = X_h, Y_vec = Y_h, dates_vec = dates_h,
        init_train_size = init_train_size, horizon = h
      )
      
      metrics_h <- calculate_metrics(forecast_output_h$results, forecast_output_h$dist_list)
      # Store metrics for final summary
      metric_row <- data.frame(
        Model        = paste0("DRF_", toupper(region)),
        horizon_step = h,
        RMSE         = metrics_h$rmse,
        MAE          = metrics_h$mae,
        CRPS         = metrics_h$crps
      )
      all_metrics <- append(all_metrics, list(metric_row))
      forecast_output_h$metrics <- metrics_h
      
      if (save_results) {
        # Save RDS distribution file (one per horizon)
        save_forecast_distributions(forecast_output_h$dist_list, region, h, CONFIG$out_dir)
        
        # Collect the results data frame for this horizon
        region_results_df_list <- append(region_results_df_list, list(forecast_output_h$results))
      }
      
      region_results_list[[paste0("h", h)]] <- forecast_output_h
    }
    
    # --- AGGREGATE AND SAVE FOR THE REGION ---
    # After all horizons for a region are done, combine and save its CSV
    if (save_results && length(region_results_df_list) > 0) {
      combined_region_forecasts <- dplyr::bind_rows(region_results_df_list)
      output_path <- file.path(CONFIG$out_dir, sprintf("%s_drf_forecast_results.csv", region))
      write_csv(combined_region_forecasts, output_path)
      
      cat(sprintf("\nSaved combined forecast results for %s to: %s\n", toupper(region), output_path))
    }
    
    full_forecast_output[[region]] <- region_results_list
  }
  
  # Combine all metrics and save to a single CSV
  if (save_results && length(all_metrics) > 0) {
    final_metrics_df <- dplyr::bind_rows(all_metrics)
    metrics_output_path <- file.path(CONFIG$out_dir, "drf_evaluation_summary.csv")
    write_csv(final_metrics_df, metrics_output_path)
    
    cat(sprintf("\nSaved combined evaluation metrics to: %s\n", metrics_output_path))
  }
  
  if (save_results) cat("\nAll files saved to:", CONFIG$out_dir, "\n")
  
  invisible(list(forecast_output = full_forecast_output, config_used = CONFIG))
}

# ---- 3. Execute DRF Analysis ----------------------------------------------
if (interactive()) {
  results <- run_drf_analysis(save_results = TRUE)
  
  # You can now access results like:
  # results$forecast_output$ch$h1$results  # Data frame of results for CH, h=1
  # results$forecast_output$us$h12$metrics # Metrics for US, h=12
}