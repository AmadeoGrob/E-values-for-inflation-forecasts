import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from numpy.linalg import svd, eigvalsh


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


# Function to get optimal parameters for current data size
def get_dfm_params(training_data_size, precomputed_params):
    """
    Selects optimal DFM parameters from a pre-computed table based on training data size.

    Args:
        training_data_size (int): The current number of time series points (T).
        precomputed_params (pd.DataFrame): A table with a 'T' column and a 'static_factors' column.

    Returns:
        int: The optimal number of static factors (k_factors).
    """
    # Find all parameter sets that were determined on a smaller or equal training set
    valid_params = precomputed_params[precomputed_params["T"] <= training_data_size]

    if valid_params.empty:
        best_params_row = precomputed_params.loc[precomputed_params["T"].idxmin()]
    else:
        best_params_row = valid_params.loc[valid_params["T"].idxmax()]

    # Get the number of static factors from the selected row
    optimal_k = int(best_params_row["static_factors"])

    # The VAR lag order 'p' is fixed at 1
    factor_order = 1

    return optimal_k, factor_order


def onatski_criterion(eigenvalues, n_max=8):
    """
    Translates the Onatski (2010) criterion for selecting the number of factors
    from a MATLAB implementation.

    This criterion identifies a "sharp drop" in the sorted eigenvalues to distinguish
    between factors and noise.

    Args:
        eigenvalues (np.ndarray or list): A 1D array or list of eigenvalues,
                                          sorted in descending order.
        n_max (int, optional): The maximum number of factors to consider.
                               Defaults to 15, as in the source code.

    Returns:
        int: The estimated number of factors (NF_ON).

    Raises:
        ValueError: If the number of eigenvalues is insufficient for the calculation.
    """
    # Ensure eigenvalues is a 1D numpy array
    ev = np.asarray(eigenvalues).flatten()

    # Parameters
    n_reg = 10  # Regression window size

    # --- Input Validation ---
    if len(ev) < n_max + n_reg:
        raise ValueError(
            f"Insufficient eigenvalues. Need at least {n_max + n_reg} "
            f"(n_max + n_reg), but got {len(ev)}."
        )

    # --- Initialization ---
    numit = 0  # Iteration counter
    j = n_max + 1  # Initial guess for the number of factors + 1
    r_hat_old = n_max  # Initialize previous estimate
    cond = True  # Loop condition

    while cond:
        numit += 1

        # --- Regression Setup ---
        # Select the dependent variable: a slice of n_reg eigenvalues starting from j
        y = ev[j - 1 : j - 1 + n_reg]

        # Create the independent variables (design matrix)
        # 1. A column of ones for the intercept
        intercept = np.ones(n_reg)
        # 2. The regressor based on the index, raised to the power of 2/3
        regressor_indices = np.arange(j - 1, j - 1 + n_reg)
        regressor = regressor_indices ** (2 / 3)

        # Combine into the design matrix 'x'
        x = np.column_stack([intercept, regressor])

        # --- OLS Estimation ---
        b = np.linalg.lstsq(x, y, rcond=None)[0]

        # --- Threshold Calculation ---
        # The threshold 'delta' is twice the absolute value of the slope coefficient
        delt = 2 * np.abs(b[1])

        # --- Determine Candidate Number of Factors ---
        # Find all factors where the drop to the next eigenvalue is greater than delta
        differences = ev[:n_max] - ev[1 : n_max + 1]
        is_significant = differences > delt

        # Create a vector of factor numbers that meet the criterion
        # If is_significant is False (0), the result is 0. If True (1), it's the index.
        r_hat_v = np.arange(1, n_max + 1) * is_significant

        # The new estimate is the largest number of factors that met the criterion
        r_hat = np.max(r_hat_v)

        # --- Loop Termination Condition ---
        # Stop if the estimate has converged or if we've iterated too many times
        if r_hat == (j - 1) or numit > 10:
            cond = False

        # Update for the next iteration
        r_hat_old = j - 1
        j = int(r_hat) + 1

    # If the loop timed out, take the minimum of the last two estimates
    if numit > 10:
        r_hat = min(r_hat, r_hat_old)

    # --- Final Logic to Determine Output ---
    if r_hat == 0:
        NF_ON = 1
    elif r_hat >= n_max:
        NF_ON = 1
    else:
        NF_ON = int(r_hat)

    return NF_ON


# Bai and Ng (2002) Information Criteria for Factor Selection
def bai_ng_criteria(data, max_factors=10):
    """
    Calculate Bai and Ng (2002) information criteria for static factor selection.

    Parameters:
    -----------
    data : pd.DataFrame
        Standardized data matrix (T x N)
    max_factors : int
        Maximum number of factors to consider

    Returns:
    --------
    dict : Dictionary containing all criteria and optimal factor numbers
    """

    # Ensure data is standardized
    if not hasattr(data, "mean") or abs(data.mean().mean()) > 1e-10:
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
    else:
        X = data.values if hasattr(data, "values") else data

    T, N = X.shape
    NT = N * T

    # Storage for criteria
    criteria_results = {
        "factors": list(range(1, max_factors + 1)),
        "ICp1": [],
        "ICp2": [],
        "PCp1": [],
        "PCp2": [],
        "sum_squared_residuals": [],
    }

    for k in range(1, max_factors + 1):
        # Perform PCA with k factors
        pca = PCA(n_components=k)
        factors = pca.fit_transform(X)  # T x k
        loadings = pca.components_.T  # N x k (loadings matrix)

        # Reconstruct data: X_hat = F * Λ'
        X_reconstructed = factors @ loadings.T

        # Calculate residuals
        residuals = X - X_reconstructed

        # Sum of squared residuals
        V_k = np.sum(residuals**2) / NT
        criteria_results["sum_squared_residuals"].append(V_k)

        # Estimate error variance (σ²)
        sigma_squared = V_k  # Simplified estimate

        # Calculate penalty terms
        penalty_NT = (N + T - k) / NT
        penalty_log_NT = penalty_NT * np.log(NT / (N + T - k))
        penalty_log_min_NT = penalty_NT * np.log(min(N, T))

        # Bai-Ng Information Criteria
        ICp1 = np.log(V_k) + k * penalty_log_NT
        ICp2 = np.log(V_k) + k * penalty_log_min_NT

        # Bai-Ng Penalty Criteria
        PCp1 = V_k + k * sigma_squared * penalty_log_NT
        PCp2 = V_k + k * sigma_squared * penalty_log_min_NT

        criteria_results["ICp1"].append(ICp1)
        criteria_results["ICp2"].append(ICp2)
        criteria_results["PCp1"].append(PCp1)
        criteria_results["PCp2"].append(PCp2)

    # Find optimal number of factors for each criterion
    optimal_factors = {
        "ICp1": np.argmin(criteria_results["ICp1"]) + 1,
        "ICp2": np.argmin(criteria_results["ICp2"]) + 1,
        "PCp1": np.argmin(criteria_results["PCp1"]) + 1,
        "PCp2": np.argmin(criteria_results["PCp2"]) + 1,
    }

    return criteria_results, optimal_factors


def dynamic_factors(X, NF, p, max_q=None, scaler=None):
    """
    Bai–Ng (2007) for dynamic factors q1 and q2.
    Parameters:
    -----------
    X : array-like, shape (T, N)
        Standardized data matrix (T x N)
    NF : int
        Number of static factors
    p : int
        VAR lag order for static factors
    max_q : int, optional
        Maximum number of dynamic factors to consider. If None, set to NF.
    scaler : StandardScaler, optional
        Pre-fitted scaler for standardization. If None, a new scaler will be fitted.
    """
    # for NF = 1, return q1 = q2 = 1, no VAR can be estimated
    if NF == 1:
        return 1, 1

    X = np.asarray(X, float)
    T, N = X.shape
    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=False).fit(X)
    Xs = scaler.transform(X)

    if max_q is None:
        max_q = NF = int(NF)
    max_q = int(min(max_q, NF))

    # Static factors via SVD (take NF)
    U, S, Vt = svd(Xs, full_matrices=False)
    F = U[:, :NF] * S[:NF]  # (T, NF)

    # VAR(p) on factors, no constant
    res = VAR(F).fit(p, trend="n")
    e = res.resid
    R = e.T @ e
    RE = np.sort(eigvalsh(R))[::-1]

    m = min(max_q, RE.size)
    if m <= 1 or np.sum(RE[:m] ** 2) <= 0:
        return 1, 1
    denom = np.sum(RE[:m] ** 2)

    D1 = np.array([(RE[lag] ** 2) / denom for lag in range(1, m)])
    D2 = np.array([np.sum(RE[lag:m] ** 2) / denom for lag in range(1, m)])

    tau = 1.0 / (min(N**0.4, T**0.4))
    k1 = np.sqrt(D1) < tau
    k2 = np.sqrt(D2) < tau

    q1 = int(np.argmax(k1) + 1) if k1.any() else 1
    q2 = int(np.argmax(k2) + 1) if k2.any() else 1
    return q1, q2
