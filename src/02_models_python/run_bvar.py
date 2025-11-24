"""
run_bvar.py - Generates rolling BVAR forecasts for multiple regions and priors.

This script performs the following steps:
1.  Loads the levels and transformed/winsorized datasets for CH, EU, and US.
2.  For each region, it runs a rolling-window forecast simulation.
3.  This is repeated for three different prior specifications:
    - Minnesota
    - Diffuse
    - Normal-Wishart
4.  The resulting forecast dataframes are saved to `results/forecasts/bvar/`.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Project Structure Setup ---
# Assumes this script is in `src/02_models_python/`.
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# --- Local Imports ---
# Assumes the BVAR helper files are in `src/bvar_utils/`
from src.bvar_utils.bvar_analyt import bvar_analyt
from src.bvar_utils.helpers import preprocess_multivar

# --- Global Configuration ---
DATA_DIR = ROOT_DIR / "data/processed"
OUTPUT_DIR = ROOT_DIR / "results/forecasts/bvar"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model & Data Configuration ---
REGIONS = ["ch", "eu", "us"]
HORIZONS = [1, 2, 3, 6, 12]

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

INFLATION_VARS_LEVELS = {"ch": "cpi_total_yoy", "eu": "hcpi_yoy", "us": "cpi_all_yoy"}

INFLATION_VARS_TRANSFORMED = {
    "ch": "cpi_total_yoy_t",
    "eu": "hcpi_yoy_t",
    "us": "cpi_all_yoy_t",
}

INITIAL_WINDOWS = {"ch": 202, "eu": 115, "us": 312}


def load_all_data():
    """Loads and preprocesses all required datasets for all regions."""
    print("--- Loading and Preprocessing Data ---")
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

    print("-" * 35)
    return multivar_data_levels, multivar_data_transformed


def run_rolling_forecast(region, data_levels, data_transformed, prior_config):
    """
    Performs the core rolling forecast logic for a single region and prior.
    """
    prior_name = prior_config["name"]
    print(f"\nProcessing {region.upper()} with {prior_name} Prior...")

    T_total = data_transformed.shape[0]
    initial_window = INITIAL_WINDOWS[region]
    inflation_var_levels = INFLATION_VARS_LEVELS[region]
    inflation_var_transformed = INFLATION_VARS_TRANSFORMED[region]

    try:
        inflation_col_idx = data_transformed.columns.get_loc(inflation_var_transformed)
    except KeyError:
        print(
            f"FATAL: Inflation variable '{inflation_var_transformed}' not found. Skipping."
        )
        return

    rows = []

    for t in tqdm(
        range(initial_window, T_total),
        desc=f"Rolling Forecast ({region.upper()}-{prior_name})",
    ):
        cutoff_date = data_transformed.index[t - 1]
        Y_train_df = data_transformed.iloc[:t, :]

        try:
            result = bvar_analyt(
                Y_train_df,
                forecasting=True,
                horizon_h=max(HORIZONS),
                forecast_method=1,
                **prior_config["params"],
            )

            predicted_changes_path = result["pred_mean"]
            predicted_var_path = result["pred_var"]
            last_observed_level = data_levels.loc[cutoff_date, inflation_var_levels]

            cumulative_changes = predicted_changes_path[:, inflation_col_idx].cumsum()
            level_forecasts = last_observed_level + cumulative_changes

            cumulative_variances = np.array(
                [
                    np.sum(
                        [
                            predicted_var_path[i, inflation_col_idx, inflation_col_idx]
                            for i in range(h)
                        ]
                    )
                    for h in range(1, len(HORIZONS) + 1)
                    if h <= predicted_var_path.shape[0]
                ]
            )

            for h in HORIZONS:
                target_idx = t - 1 + h
                if target_idx >= T_total:
                    continue

                target_time = data_transformed.index[target_idx]
                mean_val_levels = level_forecasts[h - 1]
                y_true_levels = data_levels.loc[target_time, inflation_var_levels]

                h_idx = next(
                    (i for i, horizon in enumerate(HORIZONS) if horizon == h), None
                )
                if h_idx is not None and h_idx < len(cumulative_variances):
                    cumulative_var = cumulative_variances[h_idx]
                    std_val = np.sqrt(max(cumulative_var, 1e-9))
                else:
                    var_val = float(
                        predicted_var_path[h - 1, inflation_col_idx, inflation_col_idx]
                    )
                    std_val = np.sqrt(max(var_val, 1e-9))

                rows.append(
                    {
                        "cutoff": cutoff_date,
                        "target_time": target_time,
                        "horizon_step": h,
                        "prediction": mean_val_levels,
                        "h_step_sd": std_val,
                        "y_true": y_true_levels,
                    }
                )

        except Exception as e:
            print(f"ERROR at timestep {t} for {region.upper()}-{prior_name}: {e}")
            continue

    if not rows:
        print(f"No results generated for {region.upper()} with {prior_name} prior.")
        return

    summary_df = pd.DataFrame(rows)

    filename_base = f"{region}_bvar_{prior_name.lower().replace('-', '')}"
    out_csv = OUTPUT_DIR / f"{filename_base}.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved forecasts to {out_csv.name}")


def main():
    """
    Main function to orchestrate the BVAR forecasting process.
    """
    prior_configs = [
        {
            "name": "Minnesota",
            "params": {
                "constant": False,
                "p_lags": 1,
                "prior_type": 2,
                "a_bar_1": 0.3,
                "a_bar_2": 0.1,
                "a_bar_3": 1e2,
            },
        },
        {
            "name": "Diffuse",
            "params": {
                "constant": True,
                "p_lags": 1,
                "prior_type": 1,
            },
        },
        {
            "name": "Normal-Wishart",
            "params": {
                "constant": True,
                "p_lags": 1,
                "prior_type": 3,
            },
        },
    ]

    multivar_data_levels, multivar_data_transformed = load_all_data()

    for region in REGIONS:
        for config in prior_configs:
            run_rolling_forecast(
                region=region,
                data_levels=multivar_data_levels[region],
                data_transformed=multivar_data_transformed[region],
                prior_config=config,
            )

    print("\n--- BVAR script finished successfully! ---")


if __name__ == "__main__":
    main()
