import pandas as pd
import numpy as np
from .bvar_analyt import bvar_analyt


# Preprocess to multivariate time series format
def preprocess_multivar(df, columns):
    dates = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    data = {}
    for col in columns:
        if col in df.columns:
            data[col] = pd.Series(df[col].values, index=dates).asfreq("ME").dropna()
        else:
            print(f"Warning: Column '{col}' not found in dataset")
    return pd.DataFrame(data).dropna()


# Calculate Bayesian Information Criterion (BIC) for lag selection
def calculate_bic(
    Y_train, p_lags, prior_type=2, constant=False, a_bar_1=0.5, a_bar_2=0.7, a_bar_3=1e2
):
    """
    Calculate the Bayesian Information Criterion (BIC) for a given lag order.

    Parameters:
    - Y_train: Training data (numpy array).
    - p_lags: Number of lags to include in the model.
    - prior_type: Type of prior (1 = Diffuse, 2 = Minnesota, 3 = Normal-Wishart).
    - constant: Whether to include a constant term in the model.
    - a_bar_1, a_bar_2, a_bar_3: Shrinkage parameters for Minnesota prior.

    Returns:
    - BIC value for the specified model.
    """
    try:
        # Call the bvar_analyt function
        result = bvar_analyt(
            Y_train,
            constant=constant,
            p_lags=p_lags,
            prior_type=prior_type,
            a_bar_1=a_bar_1,
            a_bar_2=a_bar_2,
            a_bar_3=a_bar_3,
        )

        # Extract necessary values from the result
        log_likelihood = result["log_likelihood"]
        num_params = result["num_params"]
        T = Y_train.shape[0]

        # Calculate BIC
        bic = -2 * log_likelihood + num_params * np.log(T)
        return bic
    except Exception as e:
        print(
            f"Error calculating BIC for lag {p_lags} with prior_type {prior_type}: {e}"
        )
        return np.inf
