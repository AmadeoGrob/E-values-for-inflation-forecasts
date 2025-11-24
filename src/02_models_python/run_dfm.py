"""
run_dfm.py - Generates rolling DFM forecasts for multiple regions.

It performs the following steps:
1.  Loads levels and transformed/winsorized datasets for CH, EU, and US.
2.  Loads pre-calculated optimal DFM parameters (factors and lags).
3.  For each region, it runs a rolling-window forecast using the
    Dynamic Factor Model (DFM) with dynamically selected parameters based on the
    training set size.
4.  The resulting forecast dataframes are saved to `results/forecasts/dfm/`.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ


# --- Project Structure Setup ---
# Assumes this script is in `src/02_models_python/`.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --- Local Imports ---
from src.dfm_utils.helpers import preprocess_multivar, get_dfm_params

# --- Global Configuration ---
DATA_DIR = ROOT_DIR / "data/processed"
PARAM_INPUT_PATH = ROOT_DIR / "results/tables/factor_selection.csv"
OUTPUT_DIR = ROOT_DIR / "results/forecasts/dfm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model & Data Configuration ---
REGIONS = ["ch", "eu", "us"]
HORIZONS = [1, 2, 3, 6, 12]
INITIAL_WINDOWS = {"ch": 202, "eu": 115, "us": 312}

VARIABLE_SETS = {
    "ch": [
        "cpi_total_yoy",
        "cpi_goods_cat_goods_ind",
        "cpi_goods_cat_services_ind",
        "cpi_housing_energy_ind",
        "cpi_food_nonalcoholic_beverages_ind",
        "cpi_transport_ind",
        "cpi_health_ind",
        "cpi_clothing_footwear_ind",
        "cpi_alcoholic_beverages_tobacco_ind",
        "cpi_household_furniture_furnishings_routine_maintenance_ind",
        "cpi_restaurants_hotels_ind",
        "cpi_recreation_culture_ind",
        "cpi_communications_ind",
        "cpi_education_ind",
        "mon_stat_mon_agg_m0_total_chf",
        "ppi_total_base_month_december_2020_ind",
        "ipi_total_base_month_december_2020_ind",
        "oilpricex",
    ],
    "eu": [
        "hcpi_yoy",
        "irt3m_eacc",
        "irt6m_eacc",
        "ltirt_eacc",
        "ppicag_ea",
        "ppicog_ea",
        "ppindcog_ea",
        "ppidcog_ea",
        "ppiing_ea",
        "ppinrg_ea",
        "hicpnef_ea",
        "hicpg_ea",
        "hicpin_ea",
        "hicpsv_ea",
        "hicpng_ea",
        "curr_eacc",
        "m2_eacc",
        "m1_eacc",
        "oilpricex",
    ],
    "us": [
        "cpi_all_yoy",
        "m1sl",
        "m2sl",
        "m2real",
        "busloans",
        "fedfunds",
        "tb3ms",
        "tb6ms",
        "gs1",
        "gs5",
        "gs10",
        "ppicmm",
        "oilpricex",
        "cpiappsl",
        "cpitrnsl",
        "cpimedsl",
        "cusr0000sac",
        "cusr0000sad",
        "cusr0000sas",
        "pcepi",
    ],
}

VARIABLE_SETS_T = {
    region: [f"{var}_t" for var in variables]
    for region, variables in VARIABLE_SETS.items()
}

CPI_VARS_LEVELS = {"ch": "cpi_total_yoy", "eu": "hcpi_yoy", "us": "cpi_all_yoy"}
CPI_VARS_TRANSFORMED = {
    "ch": "cpi_total_yoy_t",
    "eu": "hcpi_yoy_t",
    "us": "cpi_all_yoy_t",
}


def load_all_data():
    """Loads and preprocesses all required datasets for all regions."""
    print("--- Loading and Preprocessing DFM Data ---")
    multivar_data_levels = {}
    multivar_data_transformed = {}

    for region in REGIONS:
        levels_df = pd.read_csv(DATA_DIR / f"{region}_data_final.csv")
        multivar_data_levels[region] = preprocess_multivar(
            levels_df, VARIABLE_SETS[region]
        )

        transformed_df = pd.read_csv(DATA_DIR / f"{region}_data_transformed_win.csv")
        multivar_data_transformed[region] = preprocess_multivar(
            transformed_df, VARIABLE_SETS_T[region]
        )

        print(
            f"Loaded {region.upper()}: Levels={multivar_data_levels[region].shape}, Transformed={multivar_data_transformed[region].shape}"
        )

    dfm_params = pd.read_csv(PARAM_INPUT_PATH)
    print(f"Loaded DFM parameters from {PARAM_INPUT_PATH.name}")
    print("-" * 40)
    return multivar_data_levels, multivar_data_transformed, dfm_params


def run_rolling_forecast(region, data_levels, data_transformed, dfm_params):
    """Performs the core DFM rolling forecast for a single region."""

    print(f"\nProcessing {region.upper()} with Dynamic Factor Model...")

    T_total = data_transformed.shape[0]
    initial_window = INITIAL_WINDOWS[region]
    cpi_var_levels = CPI_VARS_LEVELS[region]
    cpi_var_transformed = CPI_VARS_TRANSFORMED[region]

    region_params = dfm_params[dfm_params["region"] == region].copy()
    region_params["T"] = (region_params["split"] * T_total).astype(int)

    forecast_results = []

    for t in tqdm(
        range(initial_window, T_total), desc=f"Rolling DFM Forecast ({region.upper()})"
    ):
        cutoff_date = data_transformed.index[t - 1]
        train_data = data_transformed.iloc[:t]

        try:
            num_factors, factor_orders = get_dfm_params(t, region_params)

            model = DynamicFactorMQ(
                endog=train_data, factors=num_factors, factor_orders=factor_orders
            )
            results = model.fit(maxiter=1000, disp=False)
            forecast_obj = results.get_forecast(steps=max(HORIZONS))

            last_observed_level = data_levels.loc[cutoff_date, cpi_var_levels]
            predicted_changes = forecast_obj.predicted_mean[cpi_var_transformed]
            cumulative_changes = predicted_changes.cumsum()
            level_forecasts = last_observed_level + cumulative_changes

            se_changes = forecast_obj.se_mean[cpi_var_transformed]
            cumulative_var = (se_changes**2).iloc[: max(HORIZONS)].cumsum()

            for h in HORIZONS:
                target_idx = t + h - 1
                if target_idx >= T_total:
                    continue

                target_date = data_transformed.index[target_idx]
                mean_forecast_levels = level_forecasts.iloc[h - 1]
                true_value_levels = data_levels.loc[target_date, cpi_var_levels]
                se_forecast_t = np.sqrt(cumulative_var.iloc[h - 1])

                forecast_results.append(
                    {
                        "cutoff": cutoff_date,
                        "target_time": target_date,
                        "horizon_step": h,
                        "prediction": mean_forecast_levels,
                        "h_step_sd": se_forecast_t,
                        "y_true": true_value_levels,
                        "num_factors": num_factors,
                        "num_lags": factor_orders,
                    }
                )

        except Exception as e:
            print(f"Error at timestep {t} for {region.upper()}: {e}")
            num_factors, factor_orders = get_dfm_params(t, region_params)
            for h in HORIZONS:
                forecast_results.append(
                    {
                        "cutoff": cutoff_date,
                        "target_time": np.nan,
                        "horizon_step": h,
                        "prediction": np.nan,
                        "h_step_sd": np.nan,
                        "y_true": np.nan,
                        "num_factors": num_factors,
                        "num_lags": factor_orders,
                    }
                )

    if not forecast_results:
        print(f"No DFM results generated for {region.upper()}.")
        return

    rolling_forecasts_df = pd.DataFrame(forecast_results)
    csv_output_path = OUTPUT_DIR / f"{region}_dfm.csv"
    rolling_forecasts_df.to_csv(csv_output_path, index=False)
    print(f"Saved forecasts for {region.upper()} to: {csv_output_path.name}")


def main():
    """Main function to orchestrate the DFM forecasting process."""
    data_levels, data_transformed, dfm_params = load_all_data()

    for region in REGIONS:
        run_rolling_forecast(
            region=region,
            data_levels=data_levels[region],
            data_transformed=data_transformed[region],
            dfm_params=dfm_params,
        )

    print("\n--- DFM script finished successfully! ---")


if __name__ == "__main__":
    main()
