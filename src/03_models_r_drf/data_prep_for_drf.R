# ============================================================
# Inflation Forecasting - Data Preparation Script
# ============================================================
# This script handles data loading, preprocessing, and creation of design matrices
# Output: Saves design matrices and metadata for DRF modeling

# ---- 0. Packages ----------------------------------------------------------
library(here)
library(readr)
library(lubridate)
library(dplyr)
library(tidyr)
library(zoo)

# ---- 1. Configuration -----------------------------------------------------
CONFIG <- list(
  # Paths
  root_dir = here::here("..", ".."),
  data_dir = file.path(here::here("..", ".."), "data", "processed"),
  prepared_data_dir = file.path(here::here("..", ".."), "data", "processed", "for_drf"),
  
  # Feature engineering parameters
  lag_max = 1,  # number of lagged regressors
  target_lag_max = 11, # number of lags for the target variable
  horizons = c(1, 2, 3, 6, 12),  # forecast horizons
  include_current = TRUE, # Include current period (L0) in design matrix
  
  # Training parameters
  init_train_pct = 0.4,  # Use 40% of each dataset for initial training
  min_train_obs = 100,
  
  # Other
  seed = 42
)

# ---- 2. Variable Sets -----------------------------------------------------
VARIABLE_SETS <- list(
  ch = c(
    "cpi_total_yoy", "cpi_goods_cat_goods_ind", "cpi_goods_cat_services_ind",
    "cpi_housing_energy_ind", "cpi_food_nonalcoholic_beverages_ind",
    "cpi_transport_ind", "cpi_health_ind", "cpi_clothing_footwear_ind",
    "cpi_alcoholic_beverages_tobacco_ind",
    "cpi_household_furniture_furnishings_routine_maintenance_ind",
    "cpi_restaurants_hotels_ind", "cpi_recreation_culture_ind",
    "cpi_communications_ind", "cpi_education_ind",
    "mon_stat_mon_agg_m0_total_chf",
    "ppi_total_base_month_december_2020_ind",
    "ipi_total_base_month_december_2020_ind", "oilpricex"
  ),
  eu = c(
    "hcpi_yoy", "irt3m_eacc", "irt6m_eacc", "ltirt_eacc", "ppicag_ea",
    "ppicog_ea", "ppindcog_ea", "ppidcog_ea", "ppiing_ea", "ppinrg_ea",
    "hicpnef_ea", "hicpg_ea", "hicpin_ea", "hicpsv_ea", "hicpng_ea",
    "curr_eacc", "m2_eacc", "m1_eacc", "oilpricex"
  ),
  us = c(
    "cpi_all_yoy", "m1sl", "m2sl", "m2real", "busloans", "fedfunds",
    "tb3ms", "tb6ms", "gs1", "gs5", "gs10", "ppicmm",
    "oilpricex", "cpiappsl", "cpitrnsl", "cpimedsl", "cusr0000sac",
    "cusr0000sad", "cusr0000sas", "pcepi"
  )
)

# ---- 3. Helper Functions --------------------------------------------------

#' Preprocess data into multivariate zoo object
preprocess_multivar <- function(df, columns) {
  if (!"date" %in% names(df)) stop("Column 'date' is missing.")
  
  df <- df %>%
    mutate(date = as.Date(ymd(date)),
           date = ceiling_date(date, "month") - days(1)) %>%
    arrange(date)
  
  # Warn about missing columns
  missing_cols <- setdiff(columns, names(df))
  if (length(missing_cols) > 0) {
    warning("Missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  present_cols <- intersect(columns, names(df))
  ts_df <- df %>% 
    select(date, all_of(present_cols))
  
  # Crop data to start where target variable becomes available
  # Search for explicit target variable names
  target_names <- c("cpi_total_yoy", "hcpi_yoy", "cpi_all_yoy")
  target_var <- intersect(target_names, names(ts_df))[1]  # Get first match
  if (target_var %in% names(ts_df)) {
    first_valid_idx <- which(!is.na(ts_df[[target_var]]))[1]
    if (!is.na(first_valid_idx)) {
      ts_df <- ts_df[first_valid_idx:nrow(ts_df), ]
      cat(sprintf("Cropped to start where %s is available: %d rows from %s to %s\n",
                  target_var, nrow(ts_df), min(ts_df$date), max(ts_df$date)))
    }
  }
  
  zoo(ts_df[, -1], order.by = ts_df$date)
}

make_xy <- function(x, y_name, lag_max = 5, horizons = c(1, 2, 3, 6, 12), include_current = TRUE, region_name = "unknown") {
  stopifnot(y_name %in% colnames(x))
  x <- x[order(index(x)), ]
  
  n_obs <- nrow(x)
  dates <- index(x)
  
  # Calculate usable range
  longest_lag <- max(lag_max, CONFIG$target_lag_max)
  first_usable <- longest_lag + 1
  # Base the number of rows on the shortest horizon to maximize data
  last_usable <- n_obs - min(horizons)
  n_usable <- last_usable - first_usable + 1
  
  if (n_usable <= 0) {
    stop(sprintf("Insufficient data: need at least %d observations", longest_lag + min(horizons) + 1))
  }
  
  # Create design matrix
  X_list <- list()
  var_names <- colnames(x)
  
  # Current period (L0) - if we include current
  if (include_current) {
    for (j in 1:ncol(x)) {
      X_list[[paste0(var_names[j], "_L0")]] <- as.numeric(x[first_usable:last_usable, j])
    }
  }
  
  # Lagged periods (L1 to lag_max)
  for (lag in 1:lag_max) {
    for (j in 1:ncol(x)) {
      X_list[[paste0(var_names[j], "_L", lag)]] <- as.numeric(x[(first_usable-lag):(last_usable-lag), j])
    }
  }
  
  # Additional lags for target variable only (up to target_lag_max)
  if (longest_lag > lag_max) {
    for (lag in (lag_max + 1):longest_lag) {
      X_list[[paste0(y_name, "_L", lag)]] <- as.numeric(x[(first_usable-lag):(last_usable-lag), y_name])
    }
  }
  
  # Combine into data frame
  X_df <- as.data.frame(X_list)
  dates_vec <- dates[first_usable:last_usable]
  
  # Add differenced features
  for (lag in 1:lag_max) {
    for (var in var_names) {
      current_lag_col <- paste0(var, "_L", lag - 1)
      previous_lag_col <- paste0(var, "_L", lag)
      if (current_lag_col %in% names(X_df) && previous_lag_col %in% names(X_df)) {
        X_df[[paste0(var, "_D", lag)]] <- X_df[[current_lag_col]] - X_df[[previous_lag_col]]
      }
    }
  }
  
  # Additional differenced periods for TARGET variable only
  if (longest_lag > lag_max) {
    for (lag in (lag_max + 1):longest_lag) {
      current_lag_col <- paste0(y_name, "_L", lag - 1)
      previous_lag_col <- paste0(y_name, "_L", lag)
      if (current_lag_col %in% names(X_df) && previous_lag_col %in% names(X_df)) {
        X_df[[paste0(y_name, "_D", lag)]] <- X_df[[current_lag_col]] - X_df[[previous_lag_col]]
      }
    }
  }
  
  # Create target columns (Y part) for multiple horizons
  Y_df <- as.data.frame(lapply(horizons, function(h) {
    y_vec <- rep(NA, n_usable)
    # Determine how many future values are actually available in the original ts
    num_valid_targets <- n_obs - (first_usable + h) + 1
    # We can only fill up to the number of rows we have for X
    num_to_fill <- min(n_usable, num_valid_targets)
    
    if (num_to_fill > 0) {
      # The start index in the original series 'x'
      target_start_idx <- first_usable + h
      y_vec[1:num_to_fill] <- as.numeric(x[target_start_idx:(target_start_idx + num_to_fill - 1), y_name])
    }
    return(y_vec)
  }))
  names(Y_df) <- paste0("target_h", horizons)
  
  # Handle missing values in features (X) ONLY
  if (sum(is.na(X_df)) > 0) {
    complete_cases_X <- complete.cases(X_df)
    X_df <- X_df[complete_cases_X, ]
    Y_df <- Y_df[complete_cases_X, ]
    dates_vec <- dates_vec[complete_cases_X, ]
  }
  
  # Calculate training size: 40% of original cleaned dataset minus longest_lag
  original_n <- n_obs
  init_train_size <- max(CONFIG$min_train_obs, round(original_n * CONFIG$init_train_pct) - longest_lag)
  
  cat(sprintf("Created %d × %d design matrix (%d initial training (40%% of %d months - %d lags))\n", 
              nrow(X_df), ncol(X_df), init_train_size, original_n, longest_lag))
  
  # Create complete design matrix
  design_matrix <- data.frame(
    date = dates_vec,
    X_df,
    Y_df
  )
  
  list(
    design_matrix = design_matrix,
    init_train_size = init_train_size,
    target_var = y_name
  )
}

#' Save design matrix and metadata to CSV files
save_design_matrix <- function(design_data, region_name, output_dir) {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # Save complete design matrix
  write_csv(design_data$design_matrix, file.path(output_dir, sprintf("%s_design_matrix.csv", region_name)))
  
  # Save metadata
  metadata <- data.frame(
    region = region_name,
    target_variable = design_data$target_var,
    n_observations = nrow(design_data$design_matrix),
    n_features = ncol(design_data$design_matrix) - 1 - length(CONFIG$horizons),  # Subtract date and all target columns
    init_train_size = design_data$init_train_size,
    date_start = min(design_data$design_matrix$date),
    date_end = max(design_data$design_matrix$date),
    lag_max = CONFIG$lag_max,
    horizons = paste(CONFIG$horizons, collapse = ", "),
    include_current = CONFIG$include_current
  )
  write_csv(metadata, file.path(output_dir, sprintf("%s_metadata.csv", region_name)))
  
  cat(sprintf("Saved design matrix for %s to CSV files\n", toupper(region_name)))
  
  list(
    design_matrix_file = file.path(output_dir, sprintf("%s_design_matrix.csv", region_name)),
    metadata_file = file.path(output_dir, sprintf("%s_metadata.csv", region_name))
  )
}

# ---- 4. Main Data Preparation Pipeline ------------------------------------

#' Run data preparation and save design matrices
prepare_data <- function() {
  # Create output directory
  if (!dir.exists(CONFIG$prepared_data_dir)) {
    dir.create(CONFIG$prepared_data_dir, recursive = TRUE)
  }
  
  # Load data
  datasets <- list(
    ch = read_csv(file.path(CONFIG$data_dir, "ch_data_final.csv"), show_col_types = FALSE),
    eu = read_csv(file.path(CONFIG$data_dir, "eu_data_final.csv"), show_col_types = FALSE),
    us = read_csv(file.path(CONFIG$data_dir, "us_data_final.csv"), show_col_types = FALSE)
  )
  
  # Process each region
  saved_files <- list()
  
  for (region in names(VARIABLE_SETS)) {
    
    # Preprocess to multivariate time series
    ts_zoo <- preprocess_multivar(datasets[[region]], VARIABLE_SETS[[region]])
    
    # Select target variable by name
    target_names <- c("cpi_total_yoy", "hcpi_yoy", "cpi_all_yoy")
    target_var <- intersect(target_names, names(ts_zoo))[1]
    
    # Check if target variable was found
    if (is.na(target_var)) {
      cat(sprintf("Warning: No target variable found for %s. Available: %s\n", 
                  toupper(region), paste(names(ts_zoo)[1:5], collapse = ", ")))
      next
    }
    
    # Create design matrix
    design_data <- make_xy(ts_zoo, target_var, CONFIG$lag_max, CONFIG$horizons, 
                           CONFIG$include_current, region)
    
    cat(sprintf("Design matrix: %d rows × %d features | Target: %s\n",
                nrow(design_data$design_matrix), 
                ncol(design_data$design_matrix) - 1 - length(CONFIG$horizons), 
                target_var))
    
    # Save to CSV files
    saved_files[[region]] <- save_design_matrix(design_data, region, CONFIG$prepared_data_dir)
  }
  
  # Save overall configuration
  config_df <- data.frame(
    parameter = names(CONFIG),
    value = sapply(CONFIG, function(x) ifelse(is.list(x), paste(x, collapse = ", "), as.character(x)))
  )
  write_csv(config_df, file.path(CONFIG$prepared_data_dir, "config.csv"))
  
  cat(sprintf("\nData preparation complete! Files saved to: %s\n", CONFIG$prepared_data_dir))
  cat("Files created per region:\n")
  cat("  - {region}_design_matrix.csv: Complete design matrix (date, target_date, features, target)\n")
  cat("  - {region}_metadata.csv: Dataset metadata\n")
  cat("  - config.csv: Configuration parameters\n")
  
  return(saved_files)
}

# ---- 5. Execute Data Preparation ------------------------------------------
# Run data preparation
files_saved <- prepare_data()

# ---- 6. Check final files ------------------------------------------------
for (region in names(files_saved)) {
  file_path <- files_saved[[region]]$design_matrix_file
  
  if (file.exists(file_path)) {
    # Create a unique name for each dataframe, e.g., "ch_df"
    new_var_name <- paste0(region, "_df")
    
    # Read the CSV and assign it to the new variable name in the global environment
    assign(
      new_var_name,
      read_csv(file_path, show_col_types = FALSE),
      envir = .GlobalEnv
    )
  } else {
    cat(sprintf("File not found for region %s: %s\n", toupper(region), file_path))
  }
}
