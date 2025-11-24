import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Union, Callable, Dict


@dataclass
class TSUtils:
    """
    Small collection of reusable time-series helpers **plus** the
    transformation codes used in Stock & Watson / FRED‑MD datasets
    and ECB Macro datasets.

    Parameters
    ----------
    date_col : str, default "date"
        Name of the column that holds the timestamps.
    """

    date_col: str = "date"

    #  Date handling

    def set_date_index(
        self,
        df: pd.DataFrame,
        *,
        inplace: bool = False,
        sort: bool = True,
    ) -> pd.DataFrame:
        """
        Convert *date_col* to a datetime index (optionally sorted).
        """
        _df = df if inplace else df.copy()

        if self.date_col not in _df.columns:
            raise KeyError(f"Column '{self.date_col}' not found.")

        _df[self.date_col] = pd.to_datetime(_df[self.date_col])
        _df.set_index(self.date_col, inplace=True)

        if sort:
            _df.sort_index(inplace=True)

        return _df

    def restore_date_column(
        self,
        df: pd.DataFrame,
        *,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Write the current index back to a column called *date_col*.
        """
        _df = df if inplace else df.copy()
        _df[self.date_col] = _df.index
        return _df

    #  Growth-rate function

    def compute_yoy_growth(
        self,
        df: pd.DataFrame,
        level_col: str,
        *,
        periods: int = 12,
        new_col: Union[str, None] = None,
        multiplier: float = 100.0,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Compute an annualised year-over-year growth rate of *level_col*.
        """
        _df = df if inplace else df.copy()

        if level_col not in _df.columns:
            raise KeyError(f"Column '{level_col}' not found.")

        col_out = new_col or f"{level_col}_yoy"
        _df[col_out] = multiplier * (_df[level_col] / _df[level_col].shift(periods) - 1)

        return _df

    # Helper function to extract oil price data consistently

    @staticmethod
    def extract_oil_price_data(
        us_data_path: str, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Extract oil price data from US dataset with consistent date handling and cropping.

        Args:
            us_data_path: Path to the US data CSV file
            end_date: End date for cropping the data

        Returns:
            DataFrame with 'date' and 'oilpricex' columns
        """
        us_data = pd.read_csv(us_data_path)

        # Dynamically identify the date column in the US data
        date_col = us_data.columns[0]  # The first column is the date column

        # Filter out rows with invalid date strings
        us_data = us_data[
            ~us_data[date_col].astype(str).str.startswith(("Transform", "TRANSFORM"))
        ]

        # Parse dates and handle errors
        us_data[date_col] = pd.to_datetime(us_data[date_col], errors="coerce")
        us_data = us_data.dropna(
            subset=[date_col]
        )  # Drop rows where date parsing failed

        # Crop to end date
        us_data = us_data[us_data[date_col] <= end_date]

        # Extract oil price data
        oil_price = us_data[[date_col, "OILPRICEx"]].rename(
            columns={date_col: "date", "OILPRICEx": "oilpricex"}
        )

        return oil_price

    # Helper function to crop dataframe by date

    @staticmethod
    def crop_dataframe_by_date(
        df: pd.DataFrame, end_date: pd.Timestamp, date_col: str = "date"
    ) -> pd.DataFrame:
        """Crop dataframe to end at specified date, with validation.

        Args:
            df: DataFrame to crop
            end_date: End date for cropping
            date_col: Name of the date column

        Returns:
            Cropped DataFrame
        """
        original_shape = df.shape

        # Ensure date column exists
        if date_col not in df.columns:
            raise KeyError(f"Date column '{date_col}' not found in DataFrame")

        # Crop the data
        df_cropped = df[df[date_col] <= end_date].copy()

        # Validate we still have data after cropping
        if df_cropped.empty:
            raise ValueError(f"No data remains after cropping to {end_date}")

        # Log the cropping results
        cropped_shape = df_cropped.shape
        rows_removed = original_shape[0] - cropped_shape[0]
        if rows_removed > 0:
            latest_date = df_cropped[date_col].max()
            print(
                f"    Cropped {rows_removed} rows. Latest date: {latest_date.strftime('%Y-%m')}"
            )

        return df_cropped

    # Helper to windsorize data

    @staticmethod
    def winsorize(
        data: pd.DataFrame | pd.Series,
        exclude: tuple[str, ...] | list[str] | set[str] = ("date"),
        lower: float = 0.005,
        upper: float = 0.995,
    ) -> pd.DataFrame | pd.Series:
        """
        Winsorize a DataFrame or Series at given lower/upper quantiles, ignoring NaNs.

        Args:
            data: DataFrame or Series to be winsorized.
            exclude: Columns to exclude from winsorization (only applicable for DataFrame).
            lower: Lower quantile for winsorization (0 < lower < 1).
            upper: Upper quantile for winsorization (0 < upper < 1).

        Returns:
            Winsorized DataFrame or Series.
        """
        # Validate parameters
        if not (0 < lower < 1) or not (0 < upper < 1) or lower >= upper:
            raise ValueError("`lower` must be < `upper`, and both must be in (0, 1).")

        if isinstance(data, pd.Series):
            # Handle Series input
            if data.isna().all():
                return data
            lo = data.quantile(lower, interpolation="linear")
            hi = data.quantile(upper, interpolation="linear")
            return data.clip(lower=lo, upper=hi)

        elif isinstance(data, pd.DataFrame):
            # Handle DataFrame input
            exclude_set = set(exclude)  # Convert exclude to a set for faster lookups
            out = data.copy()
            for col in out.select_dtypes(include=[np.number]).columns:
                if col in exclude_set:
                    continue
                out[col] = TSUtils.winsorize(out[col], lower=lower, upper=upper)
            return out

        else:
            raise TypeError("Input must be a pandas DataFrame or Series.")

    # Transformation helpers (tcode 1–7) from Stock & Watson / FRED‑MD datasets

    @staticmethod
    def _tcode_1(x: pd.Series) -> pd.Series:
        """(1) Level – no transformation (EU code: 0)."""
        return x

    @staticmethod
    def _tcode_2(x: pd.Series) -> pd.Series:
        """(2) First difference Δxₜ (EU code: 4)."""
        return x.diff()

    @staticmethod
    def _tcode_3(x: pd.Series) -> pd.Series:
        """(3) Second difference Δ²xₜ (EU code: 5)."""
        return x.diff().diff()

    @staticmethod
    def _tcode_4(x: pd.Series) -> pd.Series:
        """(4) Natural logarithm log xₜ (levels) (EU code: NA)."""
        return np.log(x)

    @staticmethod
    def _tcode_5(x: pd.Series) -> pd.Series:
        """(5) First difference of log Δ log xₜ ≈ growth rate (EU code: NA)
        0.01 ≈ 1% growth rate"""
        return np.log(x).diff()

    @staticmethod
    def _tcode_6(x: pd.Series) -> pd.Series:
        """(6) Second difference of log Δ² log xₜ (EU code: 3)."""
        return np.log(x).diff().diff()

    @staticmethod
    def _tcode_7(x: pd.Series) -> pd.Series:
        """(7) Difference of the period‑to‑period growth rate:
        Δ(xₜ / xₜ₋₁ − 1). (EU code: NA)"""
        growth = x / x.shift(1) - 1.0
        return growth.diff()

    @staticmethod
    def _tcode_8(x: pd.Series) -> pd.Series:
        """(8) 100 times the log of xₜ (US: NA, EU code: 1)."""
        return 100 * np.log(x)

    @staticmethod
    def _tcode_9(x: pd.Series) -> pd.Series:
        """(9) 100 times the first difference of log xₜ (US: NA, EU code: 2).
        1 ≈ 1% growth rate."""
        return 100 * np.log(x).diff()

    @classmethod
    def apply_tcode(
        cls,
        series: pd.Series,
        tcode: int,
        *,
        trim_na: bool = True,
    ) -> pd.Series:
        """Transform *series* according to the ``tcode``.

        Parameters
        ----------
        series : pd.Series
            Input data (must be positive for log transforms).
        tcode : int {1, …, 9}
            Transformation code.
        trim_na : bool, default True
            Drop the initial NaNs introduced by differencing.
        """
        if tcode not in cls._TCODE_FUNCS:
            raise ValueError("tcode must be an integer 1 – 9.")

        out = cls._TCODE_FUNCS[tcode](series.copy())
        return out.dropna() if trim_na else out


# Class-level dispatch table for transformation codes
TSUtils._TCODE_FUNCS: Dict[int, Callable[[pd.Series], pd.Series]] = {
    1: lambda x: TSUtils._tcode_1(x),
    2: lambda x: TSUtils._tcode_2(x),
    3: lambda x: TSUtils._tcode_3(x),
    4: lambda x: TSUtils._tcode_4(x),
    5: lambda x: TSUtils._tcode_5(x),
    6: lambda x: TSUtils._tcode_6(x),
    7: lambda x: TSUtils._tcode_7(x),
    8: lambda x: TSUtils._tcode_8(x),
    9: lambda x: TSUtils._tcode_9(x),
}
