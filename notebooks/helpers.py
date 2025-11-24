import numpy as np
import pandas as pd
from properscoring import crps_ensemble, crps_gaussian
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ForecastFanChart:
    """Fan chart plotter for forecast pickle files."""

    def __init__(self, actual_series):
        """
        Parameters
        ----------
        actual_series : pd.Series
            Actual time series data with datetime index
        """
        self.actual_series = actual_series

    def plot_from_python(
        self,
        forecast_path,
        horizon_step,
        title="Forecast Fan Chart",
        quantile_levels=None,
        color="red",
        method="normal",
        start_date=None,
    ):
        """
        Plot fan chart directly from forecast file (CSV or Pickle).

        Parameters
        ----------
        forecast_path : str or Path
            Path to the forecast file (CSV or Pickle)
        horizon_step : int
            Forecast horizon step to plot
        title : str
            Chart title
        quantile_levels : list
            Quantile levels for the fan chart bands
        color : str
            Color for forecast bands
        method : str
            'normal' for Normal distribution assumption,
            'empirical' for pre-calculated quantiles
        start_date : str or pd.Timestamp, optional
            Start date for plotting forecasts (inclusive)
        """
        if quantile_levels is None:
            quantile_levels = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]

        # Load and prepare forecast data
        if str(forecast_path).endswith(".pkl"):
            forecast_df = pd.read_pickle(forecast_path)
        elif str(forecast_path).endswith(".csv"):
            forecast_df = pd.read_csv(forecast_path)
        else:
            raise ValueError(f"Unsupported file type: {forecast_path}")
        forecast_data = forecast_df.reset_index()
        forecast_data["target_time"] = pd.to_datetime(forecast_data["target_time"])
        forecast_data = forecast_data[forecast_data["horizon_step"] == horizon_step]

        # Filter by start_date if provided
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            forecast_data = forecast_data[forecast_data["target_time"] >= start_date]

        # Remove rows with missing values based on method
        if method == "normal":
            forecast_data = forecast_data.dropna(subset=["prediction", "h_step_sd"])
        else:  # empirical method
            # Check for required quantile columns
            required_cols = []
            for q in quantile_levels:
                if q == 0.50:
                    if "median" in forecast_data.columns:
                        required_cols.append("median")
                    elif "q_0.50" in forecast_data.columns:
                        required_cols.append("q_0.50")
                else:
                    col_name = f"q_{q:.3f}".replace("0.", "")
                    if col_name in forecast_data.columns:
                        required_cols.append(col_name)

            if not required_cols:
                print(
                    f"No quantile columns found! Available columns: {forecast_data.columns.tolist()}"
                )
                return
            forecast_data = forecast_data.dropna(subset=required_cols)

        if len(forecast_data) == 0:
            print("No valid forecast data found!")
            return

        # Calculate quantiles for each forecast
        quantiles_dict = {q: [] for q in quantile_levels}
        forecast_dates = []

        for _, row in forecast_data.iterrows():
            if method == "normal" and row["h_step_sd"] > 0:
                # Use normal distribution
                dist = norm(loc=row["prediction"], scale=row["h_step_sd"])
                for q in quantile_levels:
                    quantiles_dict[q].append(dist.ppf(q))
                forecast_dates.append(row["target_time"])

            elif method == "empirical":
                # Use pre-calculated quantiles
                valid_quantiles = True
                row_quantiles = {}

                for q in quantile_levels:
                    if q == 0.50:
                        # Handle median column
                        if "median" in row and pd.notna(row["median"]):
                            row_quantiles[q] = float(row["median"])
                        elif "q_0.50" in row and pd.notna(row["q_0.50"]):
                            row_quantiles[q] = float(row["q_0.50"])
                        else:
                            print(f"Missing median/q_0.50 for {row['target_time']}")
                            valid_quantiles = False
                            break
                    else:
                        # Handle other quantiles with exact column names
                        if q == 0.025:
                            col_name = "q_0.025"
                        elif q == 0.975:
                            col_name = "q_0.975"
                        else:
                            col_name = f"q_{q:.2f}"

                        if col_name in row and pd.notna(row[col_name]):
                            row_quantiles[q] = float(row[col_name])
                        else:
                            print(
                                f"Missing quantile {col_name} for {row['target_time']}"
                            )
                            valid_quantiles = False
                            break

                if valid_quantiles:
                    for q in quantile_levels:
                        quantiles_dict[q].append(row_quantiles[q])
                    forecast_dates.append(row["target_time"])

        if not forecast_dates:
            print("No valid forecasts to plot!")
            return

        # Convert to arrays
        forecast_dates = pd.to_datetime(forecast_dates)
        q_arrays = {q: np.array(quantiles_dict[q]) for q in quantile_levels}

        # Create the plot
        plt.figure(figsize=(15, 8))

        # Plot actual data
        plt.plot(
            self.actual_series.index,
            self.actual_series.values,
            color="black",
            linewidth=2,
            label="Actual",
            zorder=3,
        )

        # Plot fan bands
        band_pairs = [
            (0.025, 0.975, 0.15),  # 95% band
            (0.1, 0.9, 0.25),  # 80% band
            (0.25, 0.75, 0.4),  # 50% band
        ]

        for low_q, high_q, alpha in band_pairs:
            if low_q in q_arrays and high_q in q_arrays:
                plt.fill_between(
                    forecast_dates,
                    q_arrays[low_q],
                    q_arrays[high_q],
                    color=color,
                    alpha=alpha,
                    label=f"{int(100 * (high_q - low_q))}% Interval",
                    zorder=1,
                )

        # Plot median forecast
        if 0.50 in q_arrays:
            plt.plot(
                forecast_dates,
                q_arrays[0.50],
                color=color,
                linestyle="--",
                linewidth=2,
                label="Median Forecast",
                zorder=2,
            )
        elif "median" in forecast_data.columns:
            median_data = forecast_data.dropna(subset=["median"])
            if len(median_data) > 0:
                plt.plot(
                    median_data["target_time"],
                    median_data["median"],
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    label="Median Forecast",
                    zorder=2,
                )
        if method == "empirical" and "mean" in forecast_data.columns:
            mean_data = forecast_data.dropna(subset=["mean"])
            if len(mean_data) > 0:
                plt.plot(
                    mean_data["target_time"],
                    mean_data["mean"],
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    label="Mean Forecast",
                    zorder=2,
                )

        # Formatting
        plt.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.title(title, fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("YoY Inflation (%)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_from_r(
        self,
        csv_path,
        horizon_step,
        title="Forecast Fan Chart",
        color="red",
        start_date=None,
    ):
        """
        Plot fan chart directly from CSV file generated by R.

        Parameters
        ----------
        csv_path : str or Path
            Path to the CSV file
        horizon_step : int
            Forecast horizon step to plot
        title : str
            Chart title
        color : str
            Color for forecast bands
        start_date : str or pd.Timestamp, optional
            Start date for plotting forecasts (inclusive)
        """
        # Load CSV data
        df = pd.read_csv(csv_path)
        df["target_time"] = pd.to_datetime(df["target_time"])
        df = df[df["horizon_step"] == horizon_step]

        # Filter by start_date if provided
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df = df[df["target_time"] >= start_date]

        # Map CSV columns to quantile levels
        quantile_map = {
            0.025: "q025",
            0.10: "q10",
            0.25: "q25",
            0.50: "q50",
            0.75: "q75",
            0.90: "q90",
            0.975: "q975",
        }

        # Plot
        plt.figure(figsize=(15, 8))

        # Actual data
        plt.plot(
            self.actual_series.index,
            self.actual_series.values,
            color="black",
            linewidth=2,
            label="Actual",
            zorder=3,
        )

        # Fan bands
        band_pairs = [(0.025, 0.975, 0.15), (0.10, 0.90, 0.25), (0.25, 0.75, 0.40)]
        for low_q, high_q, alpha in band_pairs:
            low_col = quantile_map[low_q]
            high_col = quantile_map[high_q]
            if low_col in df.columns and high_col in df.columns:
                plt.fill_between(
                    df["target_time"],
                    df[low_col],
                    df[high_col],
                    color=color,
                    alpha=alpha,
                    label=f"{int(100 * (high_q - low_q))}% Interval",
                    zorder=1,
                )

        # Median line
        if "q50" in df.columns:
            plt.plot(
                df["target_time"],
                df["q50"],
                color=color,
                linestyle="--",
                linewidth=2,
                label="Median Forecast",
                zorder=2,
            )

        # Mean line
        if "mean_vec" in df.columns:
            plt.plot(
                df["target_time"],
                df["mean_vec"],
                color=color,
                linestyle="-",
                linewidth=2,
                label="Mean Forecast",
                zorder=2,
            )

        plt.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.title(title, fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("YoY Inflation (%)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


# Function to calculate RMSE
def calculate_rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))


# Function to calculate MAE
def calculate_mae(actual, forecast):
    return np.mean(np.abs(actual - forecast))


# Function to calculate CRPS for Gaussian and empirical forecasts
def evaluate_forecasts(prediction_df, method, start_date=None):
    """
    Calculates RMSE, MAE, and CRPS for each forecast horizon.

    This function handles different forecast methods by using the appropriate
    columns and CRPS calculation.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        DataFrame containing forecast results.
    method : str
        The forecast method used. Accepted values: 'normal', 'empirical'.
    start_date : str or pd.Timestamp, optional
        The date from which to start the evaluation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with metrics (RMSE, MAE, CRPS) for each horizon_step.
    """
    # --- Pre-processing ---
    df = prediction_df.copy()
    df["target_time"] = pd.to_datetime(df["target_time"])

    if start_date:
        df = df[df["target_time"] >= pd.to_datetime(start_date)]

    horizons = sorted(df["horizon_step"].unique())
    results = []

    # --- Loop through each horizon ---
    for h in horizons:
        horizon_df = df[df["horizon_step"] == h]

        # --- Handle different forecast methods ---
        if method == "normal":
            # For 'normal' distribution forecasts (like Naive),
            # prediction is the mean and median.
            # We use crps_gaussian with h_step_sd.

            eval_point_df = horizon_df.dropna(subset=["y_true", "prediction"])
            if eval_point_df.empty:
                continue

            rmse = calculate_rmse(eval_point_df["y_true"], eval_point_df["prediction"])
            mae = calculate_mae(eval_point_df["y_true"], eval_point_df["prediction"])

            eval_prob_df = horizon_df.dropna(
                subset=["y_true", "prediction", "h_step_sd"]
            )
            if eval_prob_df.empty or eval_prob_df["h_step_sd"].le(0).any():
                crps = np.nan
            else:
                crps = np.mean(
                    [
                        crps_gaussian(actual, mu=mean, sig=std_dev)
                        for actual, mean, std_dev in zip(
                            eval_prob_df["y_true"],
                            eval_prob_df["prediction"],
                            eval_prob_df["h_step_sd"],
                        )
                    ]
                )

        elif method == "empirical":
            # For 'empirical' forecasts (like PNC), we have dedicated
            # 'mean' and 'median' columns and use crps_ensemble.
            eval_point_df = horizon_df.dropna(subset=["y_true", "mean", "median"])
            if eval_point_df.empty:
                continue

            rmse = calculate_rmse(eval_point_df["y_true"], eval_point_df["mean"])
            mae = calculate_mae(eval_point_df["y_true"], eval_point_df["median"])

            eval_prob_df = horizon_df.dropna(subset=["y_true", "last_20_values"])
            if eval_prob_df.empty:
                crps = np.nan
            else:
                crps = np.mean(
                    [
                        crps_ensemble(actual, ensemble)
                        for actual, ensemble in zip(
                            eval_prob_df["y_true"], eval_prob_df["last_20_values"]
                        )
                    ]
                )

        elif method == "drf":
            # Use specific columns for DRF
            eval_point_df = horizon_df.dropna(subset=["y_true", "mean_vec", "q50"])
            if eval_point_df.empty:
                continue
            rmse = calculate_rmse(eval_point_df["y_true"], eval_point_df["mean_vec"])
            mae = calculate_mae(eval_point_df["y_true"], eval_point_df["q50"])
            crps = np.nan  # CRPS is not calculated for DRF

        else:
            raise ValueError(
                f"Unknown method: '{method}'. Please use 'normal' or 'empirical'."
            )

        results.append({"horizon_step": h, "RMSE": rmse, "MAE": mae, "CRPS": crps})

    return pd.DataFrame(results)


# Split groups into two subgroups based on the number of elements
def split_group(group):
    midpoint = len(group) // 2
    return group[:midpoint], group[midpoint:]


# Function to plot time series for a group of variables
def plot_time_series(group, group_name, df):
    plt.figure(figsize=(15, 7))
    for col in group:
        plt.plot(df["date"], df[col], label=col)
    plt.title(f"Time Series for {group_name}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


def _nanminmax(*arrays):
    """nan-aware global min/max across multiple arrays."""
    stacked = np.concatenate([np.ravel(a.astype(float)) for a in arrays if a.size > 0])
    return np.nanmin(stacked), np.nanmax(stacked)


def _pad_limits(ymin, ymax, pad=0.05):
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return ymin, ymax
    if ymin == ymax:
        delta = 1.0 if ymax == 0 else abs(ymax) * 0.1
        return ymin - delta, ymax + delta
    span = ymax - ymin
    return ymin - pad * span, ymax + pad * span


def plot_group_side_by_side(
    group_cols,
    group_name,
    df_left,
    df_right,
    left_label="Transformed",
    right_label="Winsorized 0.5–99.5%",
):
    """Plot the same set of series side-by-side with identical y-limits."""
    # Guard: empty group
    group_cols = [c for c in group_cols if c in df_left.columns]
    if len(group_cols) == 0:
        print(f"[skip] No columns found for group: {group_name}")
        return

    # Compute common y-limits across both panels (based on all selected columns)
    left_vals = df_left[group_cols].to_numpy()
    right_vals = df_right[group_cols].to_numpy()
    ymin, ymax = _nanminmax(left_vals, right_vals)
    ymin, ymax = _pad_limits(ymin, ymax, pad=0.05)

    # Build figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), sharex=True)
    axL, axR = axes

    # Left: original transformed
    for col in group_cols:
        axL.plot(df_left["date"], df_left[col], label=col)
    axL.set_title(f"{group_name} — {left_label}", fontsize=14)
    axL.set_xlabel("Date")
    axL.set_ylabel("Value")
    axL.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axL.set_ylim(ymin, ymax)

    # Right: winsorized
    for col in group_cols:
        axR.plot(df_right["date"], df_right[col])
    axR.set_title(f"{group_name} — {right_label}", fontsize=14)
    axR.set_xlabel("Date")
    axR.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axR.set_ylim(ymin, ymax)

    # One shared legend for both panels
    handles, labels = axL.get_legend_handles_labels()
    if labels:
        fig.legend(
            handles, labels, loc="upper center", ncol=min(len(labels), 4), fontsize=9
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def create_results_table(
    df, region_name, horizon_groups, model_groups, model_name_map, float_format="{:.3f}"
):
    """
    Generates a LaTeX table with stacked, multi-column format.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the evaluation metrics with columns:
        'Model', 'horizon_step', 'RMSE', 'MAE', 'CRPS'.
    region_name : str
        The name of the region (e.g., "Switzerland (CH)").
    horizon_groups : list of list of int
        List of horizon groups, each group is a list of horizon steps.
    model_groups : dict
        Dictionary mapping group names to lists of model keys.
    model_name_map : dict
        Dictionary mapping model keys to pretty names.
    float_format : str, optional
        Format string for floating-point numbers. Default is "{:.3f}".
    """

    # --- Data Preparation  ---
    wide_df = (
        df.pivot_table(
            index="Model", columns="horizon_step", values=["RMSE", "MAE", "CRPS"]
        )
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1)
    )

    formatted_df = wide_df.copy().astype(object)
    for horizon in df["horizon_step"].unique():
        for metric in ["RMSE", "MAE", "CRPS"]:
            col = (horizon, metric)
            if col not in wide_df.columns:
                continue
            min_val = wide_df[col].min()
            formatted_df[col] = wide_df[col].apply(
                lambda val: (
                    f"\\textbf{{{float_format.format(val)}}}"
                    if val == min_val
                    else float_format.format(val) if pd.notnull(val) else "---"
                )
            )

    # --- Build the LaTeX tabular strings ---
    tabular_strings = []
    for horizons in horizon_groups:
        num_metrics = 3
        total_cols = 1 + len(horizons) * num_metrics
        header_lines = [
            f"\\begin{{tabular}}{{l {''.join(['ccc'] * len(horizons))}}}",
            "\\toprule",
            " & ".join(
                [""]
                + [
                    f"\\multicolumn{{{num_metrics}}}{{c}}{{\\textbf{{h={h}}}}}"
                    for h in horizons
                ]
            )
            + " \\\\",
            " ".join(
                [
                    f"\\cmidrule(lr){{{2 + j * num_metrics}-{4 + j * num_metrics}}}"
                    for j in range(len(horizons))
                ]
            ),
            " & ".join(
                ["\\textbf{Model}"]
                + ["\\textbf{RMSE} & \\textbf{MAE} & \\textbf{CRPS}"] * len(horizons)
            )
            + " \\\\",
            "\\midrule",
        ]
        body_lines = []
        for group_name, model_list in model_groups.items():
            body_lines.append(
                f"\\multicolumn{{{total_cols}}}{{l}}{{\\textit{{{group_name}}}}} \\\\"
            )
            for model_key in model_list:
                if model_key not in formatted_df.index:
                    continue
                pretty_name = model_name_map.get(model_key, model_key)
                row_data = [pretty_name] + [
                    formatted_df.loc[model_key, (h, m)]
                    for h in horizons
                    for m in ["RMSE", "MAE", "CRPS"]
                ]
                body_lines.append(" & ".join(row_data) + " \\\\")
            body_lines.append("\\midrule")
        if body_lines[-1] == "\\midrule":
            body_lines.pop()

        footer_lines = ["\\bottomrule", "\\end{tabular}"]
        tabular_strings.append("\n".join(header_lines + body_lines + footer_lines))

    # ---  Final Assembly ---

    # Dynamically create the label from the region name (e.g., "Switzerland (CH)" -> "ch")
    short_region = region_name.split("(")[-1].replace(")", "").lower().strip()

    final_table = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  ",
        "  % --- Stacked Tables ---",
        "  " + "\n  \\vspace{1em}\n  ".join(tabular_strings),
        "  ",
        f"  \\caption{{Forecasting performance for {region_name} across various horizons. Lower values are better for all metrics (RMSE, MAE, and CRPS).}}",
        f"  \\label{{tab:{short_region}_performance_summary}}",
        "\\end{table}",
    ]

    return "\n".join(final_table)


def add_right_strip(
    ax,
    label,
    *,
    size="7%",
    pad=0.0,
    face="#E6E6E6",
    edge="#000000",
    lw=0.8,
    fontsize=11,
    use_latex=True,
):
    """
    Append a narrow grey strip on the right with a black border and a rotated label.
    Works with Matplotlib's usetex=True. Assumes a LaTeX install + lmodern.
    """
    divider = make_axes_locatable(ax)
    strip = divider.append_axes("right", size=size, pad=pad)
    strip.set_facecolor(face)

    # black border
    for sp in strip.spines.values():
        sp.set_color(edge)
        sp.set_linewidth(lw)

    # make label LaTeX-safe and render via \textrm{...}
    if use_latex:
        safe = (
            label.replace("\\", r"\\")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("_", r"\_")
        )
        txt = r"\textrm{" + safe + "}"
    else:
        txt = label

    strip.text(0.5, 0.5, txt, rotation=-90, va="center", ha="center", fontsize=fontsize)

    strip.set_xticks([])
    strip.set_yticks([])
    strip.set_frame_on(True)
    return strip
