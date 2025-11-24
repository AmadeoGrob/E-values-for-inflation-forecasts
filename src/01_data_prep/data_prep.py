# data_prep.py
"""Pre‑processing pipelines for the Swiss, Euro‑Area and US monetary datasets.

Each function reads a raw data file, tidies variable names, computes year‑over‑year
(YoY) growth for the headline CPI/HICP series, and writes a compact `_final.csv`
into the *same folder that contains this script*. Further, it takes the cleaned datasets
and applies various transformations to achieve stationarity. Lastly, a winsorize function
is applied to mitigate the impact of outliers in the transformed datasets.
* **Uniform *date* column** across all outputs.
* **Canonical inflation labels**
  * 🇺🇸 ``cpi_all_yoy`` – YoY growth of *CPIAUCSL* (All‑items CPI, SA).
  * 🇪🇺 ``hcpi_yoy``    – YoY growth of *HICP_OV* (headline HICP).
  * 🇨🇭 ``cpi_total_yoy`` – YoY growth of *cpi_total_base_month_december_2020_index*.
* **Stable file names**
  * ``us_data_final.csv``, ``us_data_transformed.csv``, ``us_data_transformed_win.csv``
  * ``eu_data_final.csv``, ``eu_data_transformed.csv``, ``eu_data_transformed_win.csv``
  * ``ch_data_final.csv``, ``ch_data_transformed.csv``, ``ch_data_transformed_win.csv``
  All written into the directory that contains this script.
* **Data cropping**: All datasets are cropped to end in April 2025.

Prerequisites
-------------
* **pandas ≥ 1.4**
* ``utils.py`` providing various transformations and repetitive steps,
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from utils import TSUtils


# Global configuration

# Data cropping: All datasets will be cropped to end at this date
CROP_END_DATE = pd.Timestamp("2025-04-30")

# All *.final.csv files will be written to the `data/processed` directory,
# regardless of where the raw files live.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE_US = OUT_DIR / "us_data_final.csv"
OUT_FILE_EU = OUT_DIR / "eu_data_final.csv"
OUT_FILE_CH = OUT_DIR / "ch_data_final.csv"


# 1.  Switzerland:  ch_data


def preprocess_ch_data(
    in_path: str,
    additional_data_path: str,
    us_data_path: str,
    end_date: pd.Timestamp = CROP_END_DATE,
) -> pd.DataFrame:
    """Preprocess Swiss monetary data with cropping to specified end date.

    Args:
        in_path: Path to main Swiss data Excel file
        additional_data_path: Path to additional Swiss data Excel file
        us_data_path: Path to US data CSV file (for oil prices)
        end_date: End date for cropping the data

    Returns:
        Processed and cropped DataFrame
    """
    # Load the main Swiss monetary data
    df = pd.read_excel(in_path)

    # Load the additional dataset
    additional_df = pd.read_excel(additional_data_path)

    # Extract oil price data using helper function
    tools = TSUtils(date_col="date")
    oil_price = tools.extract_oil_price_data(us_data_path, end_date)

    # (a) tidy column names for both datasets
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" & ", "_")
        .str.replace(" ", "_")
        .str.replace("consumer_price_index", "cpi")
        .str.replace("producer_price_index", "ppi")
        .str.replace("index", "ind")
        .str.replace("categories", "cat")
        .str.replace("monetary_statistics", "mon_stat")
        .str.replace("monetary_aggregates", "mon_agg")
        .str.replace("swiss_national_bank", "snb")
        .str.replace("switzerland", "ch")
        .str.replace("end_of_day", "eod")
        .str.replace("swiss_exchange", "")
        .str.replace("swiss", "ch")
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )

    additional_df.columns = (
        additional_df.columns.str.strip()
        .str.lower()
        .str.replace(" & ", "_")
        .str.replace(" ", "_")
        .str.replace("producer_price_index", "ppi")
        .str.replace("import_price_index", "ipi")
        .str.replace("index", "ind")
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )

    # (b) ensure 'date' columns are of the same type
    df["date"] = pd.to_datetime(df["date"])
    additional_df["date"] = pd.to_datetime(additional_df["date"])

    # (c) merge the additional dataset and oil price with the main dataset
    df = pd.merge(df, additional_df, on="date", how="outer")
    df = pd.merge(df, oil_price, on="date", how="outer")

    # (c.1) crop the merged data to end date
    df = tools.crop_dataframe_by_date(df, end_date)

    # (d) parse & set the datetime index
    df = tools.set_date_index(df)

    # (e) compute YoY for main Swiss CPI
    yoy_source = "cpi_total_base_month_december_2020_ind"
    if yoy_source not in df.columns:
        raise KeyError(f"Column '{yoy_source}' not found after cleaning. Check naming.")
    df = tools.compute_yoy_growth(df, yoy_source)

    # (f) restore 'date' column and save
    df = tools.restore_date_column(df).rename(
        columns={
            "date": "date",
            f"{yoy_source}_yoy": "cpi_total_yoy",
        }
    )

    # (g) save
    df.to_csv(OUT_FILE_CH, index=False)
    print(
        f"[CH]   {in_path} + {additional_data_path} + Oil Price → {OUT_FILE_CH}   {df.shape}"
    )
    return df


# 2.  Euro Area:  ea_data  (two-sheet Excel file)


def preprocess_ea_data(
    in_path: str, us_data_path: str, end_date: pd.Timestamp = CROP_END_DATE
) -> pd.DataFrame:
    """Preprocess Euro Area data with cropping to specified end date.

    Args:
        in_path: Path to Euro Area Excel file
        us_data_path: Path to US data CSV file (for oil prices)
        end_date: End date for cropping the data

    Returns:
        Processed and cropped DataFrame
    """
    sheets = pd.read_excel(in_path, sheet_name=None)
    ea_data = sheets[list(sheets.keys())[0]]  # main data sheet
    ea_info = sheets[list(sheets.keys())[1]]  # codebook / info

    # Extract oil price data using helper function
    tools = TSUtils(date_col="date")
    oil_price = tools.extract_oil_price_data(us_data_path, end_date)

    # (a) select IDs 75–77, 92–103, 114–116
    selected_ids = (
        list(range(75, 78))  # interest rates
        + list(range(92, 104))  # prices (HICP, PPI, etc.)
        + list(range(114, 117))  # monetary aggregates
    )  # inclusive start, exclusive end
    mask = ea_info["ID"].isin(selected_ids)
    selected_names = ea_info.loc[mask, "Name"].tolist()

    # keep date col + selected variables
    date_col = ea_data.columns[0]  # first col holds dates
    ea_data = ea_data.loc[:, [date_col] + selected_names]  # Use .loc for slicing

    # (b) ensure 'date' column is of the same type
    ea_data.loc[:, date_col] = pd.to_datetime(
        ea_data[date_col]
    )  # Use .loc for assignment
    ea_data = ea_data.rename(columns={date_col: "date"})  # Rename the date column

    # (c) merge the oil price with the main dataset
    ea_data = pd.merge(ea_data, oil_price, on="date", how="outer")

    # (c.1) crop the merged data to end date
    ea_data = tools.crop_dataframe_by_date(ea_data, end_date, "date")

    # (d) set index & compute YoY
    ea_data = tools.set_date_index(ea_data)

    yoy_source = "HICPOV_EA"  # headline HICP overall index
    if yoy_source not in ea_data.columns:
        raise KeyError(
            f"'{yoy_source}' not present in the selected Euro‑Area columns.\n"
            "Check codebook names and adjust the 'yoy_source' variable."
        )
    ea_data = tools.compute_yoy_growth(ea_data, yoy_source)

    # (e) restore 'date' column and rename
    ea_data = tools.restore_date_column(ea_data).rename(
        columns={
            "date": "date",
            f"{yoy_source}_yoy": "hcpi_yoy",
        }
    )

    # (f) make all variable names lower-case
    ea_data.columns = ea_data.columns.str.lower()

    ea_data.to_csv(OUT_FILE_EU, index=False)
    print(f"[EA]   {in_path} + Oil Price → {OUT_FILE_EU}   {ea_data.shape}")
    return ea_data


# 3.  United States:  us_data  (FRED-MD CSV)


GROUP5_MONEY_CREDIT = [
    "M1SL",
    "M2SL",
    "M2REAL",
    "TOTRESNS",
    "NONBORRES",
    "BUSLOANS",
    "REALLN",
    "NONREVSL",
    "CONSPI",
    "DTCOLNVHFNM",
    "DTCTHFNM",
    "INVEST",
]
GROUP6_IR_EXR = [
    "FEDFUNDS",
    "CP3Mx",
    "TB3MS",
    "TB6MS",
    "GS1",
    "GS5",
    "GS10",
    "AAA",
    "BAA",
    "COMPAPFFx",
    "TB3SMFFM",
    "TB6SMFFM",
    "T1YFFM",
    "T5YFFM",
    "T10YFFM",
    "AAAFFM",
    "BAAFFM",
    "EXSZUSx",
    "EXJPUSx",
    "EXUSUKx",
    "EXCAUSx",
]
GROUP7_PRICES = [
    "PPICMM",
    "OILPRICEx",
    "CPIAUCSL",
    "CPIAPPSL",
    "CPITRNSL",
    "CPIMEDSL",
    "CUSR0000SAC",
    "CUSR0000SAD",
    "CUSR0000SAS",
    "CPIULFSL",
    "CUSR0000SA0L2",
    "CUSR0000SA0L5",
    "PCEPI",
    "DDURRG3M086SBEA",
    "DNDGRG3M086SBEA",
    "DSERRG3M086SBEA",
]
US_KEEP = GROUP5_MONEY_CREDIT + GROUP6_IR_EXR + GROUP7_PRICES


def preprocess_us_data(
    in_path: str, end_date: pd.Timestamp = CROP_END_DATE
) -> pd.DataFrame:
    """Preprocess US data with cropping to specified end date.

    Args:
        in_path: Path to US data CSV file
        end_date: End date for cropping the data

    Returns:
        Processed and cropped DataFrame
    """
    df = pd.read_csv(in_path)

    # (a) identify the date column
    first_col = df.columns[0]
    if first_col.lower() not in {"sasdate", "date"}:
        raise KeyError("Date column not found or unrecognized in the input file.")
    date_col = first_col

    # (b) drop the *Transform:* metadata row(s)
    df = df[~df[date_col].astype(str).str.startswith(("Transform", "TRANSFORM"))]

    # (c) drop any columns not in Groups 5-7
    keep_cols = [date_col] + [c for c in US_KEEP if c in df.columns]
    if not set(keep_cols).issubset(df.columns):
        raise KeyError("Some required columns are missing in the input file.")
    df = df[keep_cols]

    # (c.1) parse dates and rename the date column to 'date'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})

    tools = TSUtils(date_col="date")
    df = tools.crop_dataframe_by_date(df, end_date)

    # (d) set index & compute YoY for CPIAUCSL
    df = tools.set_date_index(df)
    df = tools.compute_yoy_growth(df, "CPIAUCSL")

    # (e.1) flatten & rename
    df = tools.restore_date_column(df).rename(
        columns={
            "date": "date",
            "CPIAUCSL_yoy": "cpi_all_yoy",
        }
    )

    # (e.2) make all variable names lower-case
    df.columns = df.columns.str.lower()

    df.to_csv(OUT_FILE_US, index=False)
    print(f"[US]   {in_path}  →  {OUT_FILE_US}   {df.shape}")
    return df


def apply_transformations(
    input_file: str, output_file: str, transformations: dict[str, int]
) -> None:
    """
    Apply transformations to the cleaned data and save the transformed file.

    Args:
        input_file: Path to the input .final.csv file.
        output_file: Path to save the transformed .csv file.
        transformations: Dictionary mapping column names to transformation codes (1–9).
    """
    # Load the cleaned data
    df = pd.read_csv(input_file)

    # Initialize TSUtils for date handling
    tools = TSUtils(date_col="date")

    # Ensure 'date' column is properly set
    df["date"] = pd.to_datetime(df["date"])
    df = tools.set_date_index(df)

    # Apply transformations
    for column, tcode in transformations.items():
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in the input file.")
        df[f"{column}_t"] = TSUtils.apply_tcode(df[column], tcode)

    # Restore 'date' column before filtering columns
    df = tools.restore_date_column(df)

    # Keep inflation columns unchanged
    inflation_columns = ["cpi_total_yoy", "hcpi_yoy", "cpi_all_yoy"]
    inflation_columns = [col for col in inflation_columns if col in df.columns]

    # Keep the 'date' column and transformed columns
    transformed_columns = [f"{column}_t" for column in transformations.keys()]
    df = df[["date"] + transformed_columns + inflation_columns]

    # Save the transformed data
    df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")


# 4.  Driver


# Driver section for level data
if __name__ == "__main__":
    print(f"Processing data with crop end date: {CROP_END_DATE.strftime('%Y-%m')}")
    print("=" * 60)

    # Define project root to construct absolute paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    preprocess_ch_data(
        PROJECT_ROOT / "data" / "raw" / "ch_data" / "swiss_monetary_data.xlsx",
        PROJECT_ROOT / "data" / "raw" / "ch_data" / "snb_ppi_import.xlsx",
        PROJECT_ROOT / "data" / "raw" / "us_data" / "current FRED-MD 2025-07.csv",
    )
    preprocess_ea_data(
        PROJECT_ROOT / "data" / "raw" / "eu_data" / "EAdata.xlsx",
        PROJECT_ROOT / "data" / "raw" / "us_data" / "current FRED-MD 2025-07.csv",
    )
    preprocess_us_data(
        PROJECT_ROOT / "data" / "raw" / "us_data" / "current FRED-MD 2025-07.csv"
    )

    print("=" * 60)
    print("All datasets processed and cropped to April 2025")


# Driver section for transformations
if __name__ == "__main__":
    print("=" * 60)
    print("Applying transformations to datasets...")

    # Specify transformations for Swiss data
    apply_transformations(
        input_file=str(OUT_FILE_CH),
        output_file=str(OUT_DIR / "ch_data_transformed.csv"),
        transformations={
            "cpi_total_base_month_december_2020_ind": 9,  # CPI total index level
            "cpi_core_cpi_total_excluding_fresh_seasonal_products_energy_fuels_ind": 9,
            "cpi_goods_cat_goods_ind": 9,
            "cpi_goods_cat_services_ind": 9,
            "cpi_housing_energy_ind": 9,
            "cpi_food_nonalcoholic_beverages_ind": 9,
            "cpi_transport_ind": 9,
            "cpi_health_ind": 9,
            "cpi_clothing_footwear_ind": 6,
            "cpi_alcoholic_beverages_tobacco_ind": 9,
            "cpi_household_furniture_furnishings_routine_maintenance_ind": 9,
            "cpi_restaurants_hotels_ind": 9,
            "cpi_recreation_culture_ind": 6,
            "cpi_communications_ind": 6,
            "cpi_education_ind": 9,
            "ppi_goods_cat_nondurable_consumer_goods_ind": 9,
            "ppi_goods_cat_durable_consumer_goods_ind": 9,
            "ppi_manufactured_goods_ind": 9,
            "ppi_goods_cat_intermediary_goods_ind": 9,
            "ppi_pharmaceutical_products_ind": 9,
            "ppi_goods_cat_energy_ind": 9,
            "ppi_goods_cat_investment_goods_ind": 9,
            "ppi_machines_ind": 9,
            "ppi_electricity_ind": 9,
            "ppi_metals_metal_semiproducts_ind": 9,
            "ppi_mineral_oil_products_ind": 6,
            "ppi_food_feed_ind": 9,
            "ppi_vehicles_vehicle_parts_ind": 9,
            "mon_stat_mon_agg_m0_total_chf": 6,
            "mon_stat_mon_agg_m1_total_chf": 6,
            "mon_stat_mon_agg_m2_total_chf": 6,
            "mon_stat_mon_agg_m3_total_chf": 6,
            "ch_yield_curve_spot_interest_rates_snb_10_year_yield": 2,
            "ch_yield_curve_spot_interest_rates_snb_1_year_yield": 2,
            "ch_repo_rates_six_snb_ch_average_rate_eod_3_month_fixing": 2,
            "ch_repo_rates_six_snb_ch_average_rate_eod_6_month_fixing": 2,
            "ppi_total_base_month_december_2020_ind": 9,
            "ipi_total_base_month_december_2020_ind": 9,
            "oilpricex": 6,
            "cpi_total_yoy": 2,  # YoY growth of cpi_total
        },
    )

    # Specify transformations for Euro Area data
    apply_transformations(
        input_file=str(OUT_FILE_EU),
        output_file=str(OUT_DIR / "eu_data_transformed.csv"),
        transformations={
            "irt3m_eacc": 2,
            "irt6m_eacc": 2,
            "ltirt_eacc": 2,
            "ppicag_ea": 9,
            "ppicog_ea": 9,
            "ppindcog_ea": 9,
            "ppidcog_ea": 9,
            "ppiing_ea": 9,
            "ppinrg_ea": 9,
            "hicpov_ea": 9,  # HICP overall index level
            "hicpnef_ea": 9,
            "hicpg_ea": 9,
            "hicpin_ea": 9,
            "hicpsv_ea": 9,
            "hicpng_ea": 9,
            "curr_eacc": 9,
            "m1_eacc": 9,
            "m2_eacc": 9,
            "oilpricex": 6,
            "hcpi_yoy": 2,  # YoY growth of HICPOV_EA
        },
    )

    # Specify transformations for US data
    apply_transformations(
        input_file=str(OUT_FILE_US),
        output_file=str(OUT_DIR / "us_data_transformed.csv"),
        transformations={
            "m1sl": 6,
            "m2sl": 6,
            "m2real": 6,
            "totresns": 6,
            "nonborres": 2,
            "busloans": 6,
            "realln": 6,
            "nonrevsl": 6,
            "conspi": 6,
            "dtcolnvhfnm": 6,
            "dtcthfnm": 6,
            "invest": 6,
            "fedfunds": 2,
            "cp3mx": 2,
            "tb3ms": 2,
            "tb6ms": 2,
            "gs1": 2,
            "gs5": 2,
            "gs10": 2,
            "aaa": 2,
            "baa": 2,
            "compapffx": 2,
            "tb3smffm": 2,
            "tb6smffm": 2,
            "t1yffm": 2,
            "t5yffm": 2,
            "t10yffm": 2,
            "aaaffm": 2,
            "baaffm": 2,
            "exszusx": 5,
            "exjpusx": 5,
            "exusukx": 5,
            "excausx": 5,
            "ppicmm": 6,
            "oilpricex": 6,
            "cpiaucsl": 9,  # CPIAUCSL level
            "cpiappsl": 9,
            "cpitrnsl": 9,
            "cpimedsl": 9,
            "cusr0000sac": 9,
            "cusr0000sad": 9,
            "cusr0000sas": 9,
            "cpiulfsl": 9,
            "cusr0000sa0l2": 9,
            "cusr0000sa0l5": 9,
            "pcepi": 9,
            "ddurrg3m086sbea": 9,
            "dndgrg3m086sbea": 9,
            "dserrg3m086sbea": 9,
            "cpi_all_yoy": 2,  # YoY growth of CPIAUCSL
        },
    )

    print("=" * 60)
    print("All datasets transformed and saved.")


# Driver section for winsorization
if __name__ == "__main__":
    print("=" * 60)
    print("Applying winsorization to transformed datasets...")

    # Winsorize Swiss data
    swiss_exclude = ["date", "cpi_total_yoy", "cpi_total_yoy_t"]
    swiss_input = OUT_DIR / "ch_data_transformed.csv"
    swiss_output = OUT_DIR / "ch_data_transformed_win.csv"
    swiss_df = pd.read_csv(swiss_input)
    swiss_df = TSUtils.winsorize(swiss_df, exclude=swiss_exclude)
    swiss_df.to_csv(swiss_output, index=False)
    print(f"Winsorized Swiss data saved to {swiss_output}")

    # Winsorize Euro Area data
    euro_exclude = ["date", "hcpi_yoy", "hcpi_yoy_t"]
    euro_input = OUT_DIR / "eu_data_transformed.csv"
    euro_output = OUT_DIR / "eu_data_transformed_win.csv"
    euro_df = pd.read_csv(euro_input)
    euro_df = TSUtils.winsorize(euro_df, exclude=euro_exclude)
    euro_df.to_csv(euro_output, index=False)
    print(f"Winsorized Euro Area data saved to {euro_output}")

    # Winsorize US data
    us_exclude = ["date", "cpi_all_yoy", "cpi_all_yoy_t"]
    us_input = OUT_DIR / "us_data_transformed.csv"
    us_output = OUT_DIR / "us_data_transformed_win.csv"
    us_df = pd.read_csv(us_input)
    us_df = TSUtils.winsorize(us_df, exclude=us_exclude)
    us_df.to_csv(us_output, index=False)
    print(f"Winsorized US data saved to {us_output}")

    print("=" * 60)
    print("All transformed datasets winsorized and saved.")
