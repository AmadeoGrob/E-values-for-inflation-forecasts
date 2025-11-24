# ---- Helper Functions --------------------------------------------------

#' Loads data with multiple target columns
load_design_matrix <- function(region_name, data_dir) {
  design_matrix <- read_csv(file.path(data_dir, sprintf("%s_design_matrix.csv", region_name)), show_col_types = FALSE)
  metadata <- read_csv(file.path(data_dir, sprintf("%s_metadata.csv", region_name)), show_col_types = FALSE)
  
  target_cols <- grep("^target_h", names(design_matrix), value = TRUE)
  feature_cols <- setdiff(names(design_matrix), c("date", target_cols))
  
  X <- as.data.frame(design_matrix[, feature_cols])
  Y_df <- as.data.frame(design_matrix[, target_cols])
  dates <- as.Date(design_matrix$date)
  
  # Horizons are read directly from metadata for consistency
  horizons <- as.numeric(unlist(strsplit(metadata$horizons, ", ")))
  
  cat(sprintf("Loaded %s: %d rows × %d features | Horizons: %s\n",
              toupper(region_name), nrow(X), ncol(X), paste(horizons, collapse=", ")))
  
  list(
    X = X,
    Y_df = Y_df,
    dates = dates,
    horizons = horizons,
    metadata = metadata
  )
}

#' Store distribution weights and values
store_dist <- function(w_row, y_train) {
  nz <- which(w_row > 0)
  list(idx = nz, w = w_row[nz], y = y_train[nz])
}

#' Fit DRF model with consistent parameters
fit_drf <- function(X, Y) {
  set.seed(CONFIG$seed)
  drf(
    X = X, Y = Y,
    num.trees = CONFIG$num_trees,
    splitting.rule = CONFIG$splitting_rule,
    sample.fraction = CONFIG$sample_fraction,
    min.node.size = CONFIG$min_node_size
  )
}

#' Calculate and plot variable importance per horizon
analyze_variable_importance <- function(X, Y_vec, region_name, horizon) {
  set.seed(CONFIG$seed)
  fit <- fit_drf(X, Y_vec)
  vip <- suppressWarnings(variable_importance(fit))
  names(vip) <- colnames(X)
  top <- sort(vip, decreasing = TRUE)[1:CONFIG$vip_top_n]
  
  p <- ggplot(data.frame(var = factor(names(top), levels = names(top)), imp = top),
              aes(x = reorder(var, imp), y = imp)) +
    geom_col() + coord_flip() +
    labs(x = NULL, y = "MMD importance",
         title = sprintf("Top %d Predictors (%s, h=%d)", CONFIG$vip_top_n, toupper(region_name), horizon)) +
    theme_minimal(base_size = 11)
  print(p)
}

#' Perform rolling-origin forecasting for a single region and horizon
rolling_forecast <- function(X_df, Y_vec, dates_vec, init_train_size, horizon) {
  set.seed(CONFIG$seed)
  
  n_tot <- nrow(X_df)
  n_fore <- n_tot - init_train_size
  
  pred_mat <- matrix(NA_real_, n_fore, length(CONFIG$quantiles), dimnames = list(NULL, CONFIG$quantile_names))
  dist_list <- vector("list", n_fore)
  mean_vec <- numeric(n_fore)
  cutoff_vec <- as.Date(rep(NA, n_fore))
  target_time_vec <- as.Date(rep(NA, n_fore))
  
  for (i in seq_len(n_fore)) {
    if (i %% 50 == 0) cat(sprintf("  h=%d progress: %d/%d\n", horizon, i, n_fore))
    
    idx_tr <- 1:(init_train_size + i - 1)
    idx_pred <- init_train_size + i
    
    # The date of the features used to make the forecast
    cutoff_vec[i] <- dates_vec[idx_pred]
    # The date that the forecast is for
    target_time_vec[i] <- dates_vec[idx_pred] %m+% months(horizon)
    
    fit_i <- fit_drf(X_df[idx_tr, ], Y_vec[idx_tr])
    raw <- predict(fit_i, newdata = X_df[idx_pred, , drop = FALSE])
    
    w_i <- as.numeric(raw$weights[1, ])
    mean_vec[i] <- sum(w_i * raw$y)
    dist_list[[i]] <- store_dist(w_i, raw$y)
    pred_mat[i, ] <- predict(fit_i, newdata = X_df[idx_pred, , drop = FALSE],
                             functional = "quantile", quantiles = CONFIG$quantiles)$quantile
  }
  
  results <- data.frame(
    cutoff = cutoff_vec,
    target_time = target_time_vec,
    horizon_step = horizon,
    y_true = Y_vec[(init_train_size + 1):n_tot]
  )
  results <- cbind(results, pred_mat, data.frame(mean_vec = mean_vec))
  
  list(results = results, dist_list = dist_list)
}

#' Calculate and report key performance metrics
calculate_metrics <- function(results, dist_list) {
  # Ensure no NA values in truth or predictions before calculating
  valid_rows <- complete.cases(results)
  if (sum(valid_rows) == 0) {
    return(list(rmse = NA, mae = NA, crps = NA))
  }
  
  res_valid <- results[valid_rows, ]
  dist_valid <- dist_list[valid_rows]
  
  rmse <- sqrt(mean((res_valid$y_true - res_valid$mean_vec)^2))
  mae <- mean(abs(res_valid$y_true - res_valid$q50))
  
  crps_vec <- sapply(seq_len(nrow(res_valid)), function(i) {
    d <- dist_valid[[i]]
    crps_sample(y = res_valid$y_true[i], dat = d$y, w = d$w, method = "edf")
  })
  crps <- mean(crps_vec, na.rm = TRUE)
  
  cat(sprintf("  Metrics -> RMSE: %.3f | MAE: %.3f | CRPS: %.3f\n", rmse, mae, crps))
  list(rmse = rmse, mae = mae, crps = crps)
}

#' Save forecast distribution to RDS file
save_forecast_distributions <- function(dist_list, region_name, horizon, output_dir) {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # Only save the distributions here
  saveRDS(dist_list, file.path(output_dir, sprintf("%s_h%d_distributions.rds", region_name, horizon)))
  
  cat(sprintf("  Saved distribution for %s (h=%d)\n", toupper(region_name), horizon))
}
