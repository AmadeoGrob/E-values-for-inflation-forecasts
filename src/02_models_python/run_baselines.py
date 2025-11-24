# =============================================================================
# BASELINE MODELS - FORECAST GENERATION SCRIPT
# =============================================================================
# This script generates forecasts for four baseline models:
# 1. Naive ("last"): Predicts the last observed value.
# 2. Naive ("mean"): Predicts the mean of the last 20 observations.
# 3. PNC: Creates an empirical distribution from the last 20 observations.
# 4. ARIMA(1,1,0): A fixed-order ARIMA model.
# =============================================================================

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. SETUP: PATHS AND CONFIGURATION ---------------------------------------

# This script assumes it is located in: /src/02_models_python/
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Define the standardized input and output directories
DATA_DIR_PROCESSED = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results" / "forecasts" / "baseline"

# Create the output directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 2. DATA PREPARATION -----------------------------------------------------


def preprocess_to_series(df, column):
    """Converts a DataFrame column to a clean, monthly-frequency pandas Series."""
    dates = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return pd.Series(df[column].values, index=dates).asfreq("ME").dropna()


# --- 3. MODEL-SPECIFIC FORECASTING FUNCTIONS ---------------------------------


def generate_naive_last_forecasts(series_data):
    print("\n--- Generating Naive ('last') forecasts ---")
    regions = ["ch", "eu", "us"]
    horizons = [1, 2, 3, 6, 12]
    for region in regions:
        df = series_data[region]
        predictions = []
        for i in range(len(df)):
            cutoff, prediction_val = df.index[i], df.iloc[i]
            for h in horizons:
                target_time = cutoff + pd.offsets.MonthEnd(h)
                y_true = df.loc[target_time] if target_time in df.index else None
                predictions.append(
                    {
                        "cutoff": cutoff,
                        "target_time": target_time,
                        "horizon_step": h,
                        "prediction": prediction_val,
                        "y_true": y_true,
                    }
                )

        pred_df = pd.DataFrame(predictions)
        pred_df["squared_residual"] = (pred_df["prediction"] - pred_df["y_true"]) ** 2
        pred_df["sigma_hat"] = pred_df.groupby("horizon_step")[
            "squared_residual"
        ].transform(
            lambda x: np.sqrt(x.expanding().sum() / (x.expanding().count() - 1)).shift(
                1
            )
        )
        pred_df["h_step_sd"] = pred_df["sigma_hat"]

        output_path = RESULTS_DIR / f"{region}_naive_last.csv"
        pred_df.to_csv(output_path, index=False)
        print(f"Saved {region.upper()} Naive ('last') forecasts to: {output_path}")


def generate_naive_mean_forecasts(series_data, window=20):
    print("\n--- Generating Naive ('mean') forecasts ---")
    regions = ["ch", "eu", "us"]
    horizons = [1, 2, 3, 6, 12]
    for region in regions:
        df = series_data[region]
        predictions = []
        for i in range(window, len(df)):
            cutoff = df.index[i]
            prediction_val = df.iloc[i - window : i].mean()
            for h in horizons:
                target_time = cutoff + pd.offsets.MonthEnd(h)
                y_true = df.loc[target_time] if target_time in df.index else None
                predictions.append(
                    {
                        "cutoff": cutoff,
                        "target_time": target_time,
                        "horizon_step": h,
                        "prediction": prediction_val,
                        "y_true": y_true,
                    }
                )

        pred_df = pd.DataFrame(predictions)
        pred_df["squared_residual"] = (pred_df["prediction"] - pred_df["y_true"]) ** 2

        def calc_sigma(x):
            denom = x.expanding().count() - 1 - window
            var = x.expanding().sum() / denom
            var[denom <= 0] = np.nan
            return np.sqrt(var).shift(1)

        pred_df["sigma_hat"] = pred_df.groupby("horizon_step")[
            "squared_residual"
        ].transform(calc_sigma)
        pred_df["h_step_sd"] = pred_df["sigma_hat"] * np.sqrt(1 + 1 / window)

        output_path = RESULTS_DIR / f"{region}_naive_mean.csv"
        pred_df.to_csv(output_path, index=False)
        print(f"Saved {region.upper()} Naive ('mean') forecasts to: {output_path}")


def generate_pnc_forecasts(series_data, window=20):
    """
    Generates and saves forecasts for the PNC model, including all summary statistics.
    """
    print("\n--- Generating PNC forecasts ---")
    regions = ["ch", "eu", "us"]
    horizons = [1, 2, 3, 6, 12]

    for region in regions:
        df = series_data[region]
        predictions = []

        for i in tqdm(range(window, len(df)), desc=f"Rolling PNC for {region.upper()}"):
            cutoff = df.index[i]
            last_n_obs = df.iloc[i - window : i].dropna().to_numpy()

            # Calculate all summary statistics
            stats = {}
            if last_n_obs.size > 0:
                stats = {
                    "mean": np.mean(last_n_obs),
                    "sd": np.std(last_n_obs, ddof=1),
                    "last_20_values": last_n_obs.tolist(),  # Store as a Python list
                    "q_0.025": np.percentile(last_n_obs, 2.5),
                    "q_0.10": np.percentile(last_n_obs, 10),
                    "q_0.25": np.percentile(last_n_obs, 25),
                    "median": np.percentile(last_n_obs, 50),
                    "q_0.75": np.percentile(last_n_obs, 75),
                    "q_0.90": np.percentile(last_n_obs, 90),
                    "q_0.975": np.percentile(last_n_obs, 97.5),
                }
            else:  # Handle case where window is empty
                stats = {
                    "mean": np.nan,
                    "sd": np.nan,
                    "last_20_values": [],
                    "q_0.025": np.nan,
                    "q_0.10": np.nan,
                    "q_0.25": np.nan,
                    "median": np.nan,
                    "q_0.75": np.nan,
                    "q_0.90": np.nan,
                    "q_0.975": np.nan,
                }

            for h in horizons:
                target_time = cutoff + pd.offsets.MonthEnd(h)
                y_true = df.loc[target_time] if target_time in df.index else None

                # Append a row with all original columns
                row = {
                    "cutoff": cutoff,
                    "target_time": target_time,
                    "horizon_step": h,
                    "y_true": y_true,
                    **stats,  # Unpack the entire dictionary of stats here
                }
                predictions.append(row)

        prediction_df = pd.DataFrame(predictions)

        # Save results
        output_path = RESULTS_DIR / f"{region}_pnc_model.csv"
        prediction_df.to_csv(output_path, index=False)
        print(f"Saved {region.upper()} PNC forecasts to: {output_path}")


def generate_arima_110_forecasts(series_data):
    """Generates and saves forecasts for the fixed ARIMA(1,1,0) model."""
    print("\n--- Generating ARIMA(1,1,0) forecasts ---")
    regions = ["ch", "eu", "us"]
    horizons = [1, 2, 3, 6, 12]

    for region in regions:
        df = series_data[region]
        predictions = []

        # Define the start date for forecasting (20% split)
        start_loc = int(len(df) * 0.2)
        forecast_cutoffs = df.index[start_loc:]

        for cutoff_date in tqdm(
            forecast_cutoffs, desc=f"Rolling ARIMA(1,1,0) for {region.upper()}"
        ):
            train_series = df[:cutoff_date]

            try:
                model = ARIMA(train_series, order=(1, 1, 0)).fit()
                forecast_obj = model.get_forecast(steps=max(horizons))

                pred_mean = forecast_obj.predicted_mean
                pred_se = forecast_obj.se_mean

                for h in horizons:
                    if h <= len(pred_mean):
                        target_time = pred_mean.index[h - 1]
                        predictions.append(
                            {
                                "cutoff": cutoff_date,
                                "target_time": target_time,
                                "horizon_step": h,
                                "prediction": pred_mean.iloc[h - 1],
                                "h_step_sd": pred_se.iloc[h - 1],
                                "y_true": (
                                    df.loc[target_time]
                                    if target_time in df.index
                                    else None
                                ),
                            }
                        )
            except Exception as e:
                print(f"ARIMA fit failed at {cutoff_date} for {region}: {e}")

        prediction_df = pd.DataFrame(predictions)
        output_path = RESULTS_DIR / f"{region}_arima_110.csv"
        prediction_df.to_csv(output_path, index=False)
        print(f"Saved {region.upper()} ARIMA(1,1,0) forecasts to: {output_path}")


# --- 4. MAIN ORCHESTRATOR FUNCTION -------------------------------------------


def main():
    """Main function to run the entire baseline forecasting pipeline."""
    print("--- Starting All Baseline Model Forecasts ---")

    datasets = {
        "ch": pd.read_csv(DATA_DIR_PROCESSED / "ch_data_final.csv"),
        "eu": pd.read_csv(DATA_DIR_PROCESSED / "eu_data_final.csv"),
        "us": pd.read_csv(DATA_DIR_PROCESSED / "us_data_final.csv"),
    }

    series = {
        "ch": preprocess_to_series(datasets["ch"], "cpi_total_yoy"),
        "eu": preprocess_to_series(datasets["eu"], "hcpi_yoy"),
        "us": preprocess_to_series(datasets["us"], "cpi_all_yoy"),
    }

    # Run each baseline model function
    generate_naive_last_forecasts(series)
    generate_naive_mean_forecasts(series)
    generate_pnc_forecasts(series)
    generate_arima_110_forecasts(series)

    print("\n--- Baseline Forecast Generation Complete ---")


# --- 5. SCRIPT EXECUTION -----------------------------------------------------

if __name__ == "__main__":
    main()
