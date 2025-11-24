"""
bvar_analyt.py  – Bayesian VAR estimation, forecasting, and basic posterior output
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv
from numpy import kron

# ---- local utilities ----------------------------------------------------
from .mlag2 import mlag2  # creates lag matrix
from .wish import wish  # draws samples from Wishart distribution


def bvar_analyt(
    Y_data,
    constant=True,
    p_lags=2,
    prior_type=2,
    horizon_h=1,
    forecasting=True,
    forecast_method=0,
    a_bar_1=0.3,
    a_bar_2=0.15,
    a_bar_3=10,
    sample_posterior=False,
    num_samples=1000,
):
    """
    Bayesian VAR estimation and forecasting function:
    ----------
    Parameters
    ----------
    Y_data: pd.DataFrame
        The input data for the VAR model, with observations in rows and variables in columns.
    constant: bool
        Include an intercept term in the model.
    p_lags: int
        The number of lags to include for the dependent variables.
    prior_type: int
        The type of prior to use (1 = Diffuse, 2 = Minnesota, 3 = Normal-Wishart).
    horizon_h: int
        The forecast horizon (number of steps ahead to forecast).
    forecasting: bool
        Whether to compute forecasts.
    forecast_method: int
        The method to use for forecasting (0 = direct, 1 = iterated).
    a_bar_1: float
        Shrinkage parameter for own lags (Minnesota prior).
    a_bar_2: float
        Shrinkage parameter for cross lags (Minnesota prior).
    a_bar_3: float
        Shrinkage parameter for exogenous variables (Minnesota prior).
    sample_posterior: bool
        Whether to sample from the posterior distribution.
    num_samples: int
        The number of samples to draw if sampling is enabled.
    ----------
    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "pred_mean": The predicted mean inflation
        - "pred_var": The predicted variance of inflation
        - "alpha_mean": The mean of the posterior distribution for the VAR coefficients
        - "alpha_var": The variance of the posterior distribution for the VAR coefficients
        - "A_post": The posterior VAR coefficients
        - "SIGMA_post": The posterior residual covariance matrix
        - "SIGMA_OLS": The OLS residual covariance matrix
        - "log_likelihood": The log-likelihood of the model
        - "num_params": The number of parameters in the model
        - "SIGMA_samples": The samples from the posterior distribution of the residual covariance matrix
    """

    Yraw = Y_data.copy()  # Copy input data to avoid modifying the original data
    Yraw = np.asarray(Yraw)

    # ------------------ FORECAST-SPECIFIC SPLIT AND REGRESSOR MATRIX ------------------
    T, M = Yraw.shape  # rows=time (T), cols=series (M)

    if forecasting and forecast_method == 0:
        # Direct forecasting: Target series
        Y1 = Yraw[horizon_h:, :]

        # Regressors base series: Y2 = Yraw[1 : T - h + 1, :]
        Y2 = Yraw[1 : Yraw.shape[0] - horizon_h + 1, :]

        # Build lags from Y2 so that lag-1(Y2) = original 0-lag (Y_t)
        Ylag = mlag2(Y2, p_lags)

        # Now drop first p rows (to align with lag construction)
        if constant:
            X_full = np.hstack((np.ones((Ylag.shape[0] - p_lags, 1)), Ylag[p_lags:, :]))
        else:
            X_full = Ylag[p_lags:, :]

        # Final aligned targets (drop first p rows)
        Y_full = Y1[p_lags:, :]

        # Hold out exactly one obs for evaluation
        keep = 1
        Y = Y_full[:-keep, :]
        X = X_full[:-keep, :]
        T = Y.shape[0]
    else:  # If iterated forecasting or no forecasting
        Ylag = mlag2(Yraw, p_lags)
        if constant:
            X_full = np.hstack((np.ones((T - p_lags, 1)), Ylag[p_lags:, :]))
        else:
            X_full = Ylag[p_lags:, :]
        Y_full = Yraw[p_lags:, :]

    # Sizes before hold-out
    K = X_full.shape[1]
    T_full = Y_full.shape[0]

    # --------------- HOLD-OUT SAMPLE FOR FORECAST EVALUATION ------------------------
    if forecasting:
        keep = (
            1 if forecast_method == 0 else horizon_h
        )  # set to 1 for iterated forecasts
        if T_full <= keep:
            raise ValueError(
                f"Not enough data to hold out {keep} observations for forecasting."
            )
        Y = Y_full[:-keep, :]
        X = X_full[:-keep, :]
        T = T_full - keep
    else:
        Y, X = Y_full, X_full
        T = T_full

    # Ensure T is large enough for the model, i.e. more observations than regressors
    if T < K:
        raise ValueError(
            f"Not enough observations (T={T}) to estimate the model with K={K} regressors."
        )

    # ----------------------------- OLS BASELINE ------------------------
    # Ensure arrays
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)

    # Solve instead of inverting
    XtX = X.T @ X
    XtY = X.T @ Y
    try:
        A_OLS = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        A_OLS = np.linalg.solve(XtX + 1e-6 * np.eye(XtX.shape[0]), XtY)

    a_OLS = A_OLS.ravel(order="F")
    E = Y - X @ A_OLS
    SSE = E.T @ E
    SSE = 0.5 * (SSE + SSE.T)  # numerical symmetry hygiene
    SIGMA_OLS = SSE / (T - K)  # unbiased OLS covariance

    # ------------------------ PRIOR SPECIFICATION ----------------------
    if prior_type == 1:  # Diffuse prior, no prior information used
        pass  # Posterior based purely on OLS estimates

    elif prior_type == 2:  # Minnesota
        A_prior = np.zeros((K, M))  # Prior mean
        a_prior = A_prior.ravel(order="F")

        Y_for_prior = Y

        sigma_sq = np.zeros(M)  # Residual variances for each series
        for i in range(M):
            # Build AR(p) design for series i (no constant), aligned with Y_for_prior
            y_i = Y_for_prior[p_lags:, i]
            X_i = mlag2(Y_for_prior[:, [i]], p_lags)[p_lags:, :]
            # Stable solve with small ridge
            XtX_i = X_i.T @ X_i
            XtY_i = X_i.T @ y_i
            try:
                beta_i = np.linalg.solve(XtX_i, XtY_i)
            except np.linalg.LinAlgError:
                beta_i = np.linalg.solve(XtX_i + 1e-8 * np.eye(XtX_i.shape[0]), XtY_i)
            resid_i = y_i - X_i @ beta_i
            sigma_sq[i] = float(resid_i.T @ resid_i) / max(len(y_i), 1)
        eps = 1e-12
        sigma_sq = np.maximum(sigma_sq, eps)

        V_i = np.zeros((K, M))  # Prior variance matrix for coefficients
        for eq in range(M):
            for j in range(K):
                if constant and j == 0:
                    V_i[j, eq] = a_bar_3 * sigma_sq[eq]  # intercept
                else:
                    # identify lag block and variable index
                    lag_num = (j - int(constant)) // M + 1
                    var_idx = (j - int(constant)) % M
                    if var_idx == eq:  # own lag
                        V_i[j, eq] = a_bar_1 / (lag_num**2)
                    else:  # cross lag (scale invariant)
                        V_i[j, eq] = (a_bar_2 * sigma_sq[eq]) / (
                            (lag_num**2) * sigma_sq[var_idx]
                        )

        # Safety floor
        V_i = np.maximum(V_i, 1e-12)

        # Diagonal prior precision in vec(A) order
        V_prior_inv = np.diag(1.0 / V_i.flatten(order="F"))

        # Fixed Sigma choice inside posterior:
        SIGMA_fixed_inv = np.diag(1.0 / sigma_sq)

    elif prior_type == 3:  # Normal-Wishart
        A_prior = np.zeros((K, M))
        a_prior = A_prior.ravel(order="F")  # Zero mean prior
        V_prior_inv = inv(10 * np.eye(K))  # Prior precision matrix
        v_prior = M  # Prior degrees of freedom
        S_prior = np.eye(M)  # Prior scale matrix
    else:
        raise ValueError("prior_type must be 1, 2 or 3.")

    # ------------------------- POSTERIORS ------------------------------
    if prior_type == 1:  # Diffuse, basic OLS results
        V_post = np.linalg.inv(XtX)
        A_post = A_OLS
        a_post = a_OLS
        S_post = SSE
        v_post = T - K
        df_eff = v_post - M - 1
        SIGMA_post = S_post / (df_eff)
        alpha_mean = a_post
        alpha_var = kron(SIGMA_post, V_post)

        SIGMA_samples = None

    elif prior_type == 2:  # Minnesota
        V_post = inv(
            V_prior_inv + kron(SIGMA_fixed_inv, XtX)
        )  # Posterior coefficient variance
        a_post = V_post @ (
            V_prior_inv @ a_prior + kron(SIGMA_fixed_inv, XtX) @ a_OLS
        )  # Posterior coefficient mean
        A_post = a_post.reshape(K, M, order="F")
        alpha_mean = a_post
        alpha_var = V_post
        SIGMA_post = np.diag(sigma_sq)

        SIGMA_samples = None

    elif prior_type == 3:  # Normal-Wishart
        V_post = np.linalg.inv(V_prior_inv + XtX)  # Posterior coefficient variance
        A_post = V_post @ (
            V_prior_inv @ A_prior + XtX @ A_OLS
        )  # Posterior mean as weighted average of prior mean and OLS solution
        a_post = A_post.ravel(order="F")
        S_post = (
            SSE
            + S_prior
            + A_OLS.T @ XtX @ A_OLS
            + A_prior.T @ V_prior_inv @ A_prior
            - A_post.T @ (V_prior_inv + XtX) @ A_post
        )  # Posterior scale matrix
        v_post = T + v_prior  # Posterior DoF
        df_eff = v_post - M - 1
        alpha_mean = a_post
        SIGMA_post = S_post / df_eff
        alpha_var = kron(SIGMA_post, V_post)

        if sample_posterior:
            SIGMA_samples = wish(S_post, v_post, size=num_samples)
        else:
            SIGMA_samples = None

    # -------------------- PREDICTIVE POINT FORECAST --------------------
    if forecasting:
        if forecast_method == 0:
            x_t = X_full[-1, :]  # Last available regressor row
            pred_mean = x_t @ A_post  # Forecasted mean for all M
            x_t_kron = kron(np.eye(M), x_t.reshape(1, -1))
            pred_var = (
                x_t_kron @ alpha_var @ x_t_kron.T + SIGMA_post
            )  # Forecast variance
            pred_mean_all = pred_mean
            pred_var_all = pred_var
        else:  # Iterated forecasting
            # Initialize an array to store the sequence of predicted changes
            pred_mean_all = np.zeros((horizon_h, M))
            pred_var_all = np.zeros((horizon_h, M, M))

            # Use the last `p_lags` observations from the training data as the starting point
            Y_fcst_start = Y[-p_lags:, :]

            for h in range(horizon_h):
                # Construct the regressor vector `x_t` from the most recent data
                x_t_lags = Y_fcst_start[-(p_lags):, :].ravel(order="F")
                if constant:
                    x_t = np.hstack([1.0, x_t_lags])
                else:
                    x_t = x_t_lags

                # Predict the next 1-step change
                pred_mean_h = x_t @ A_post

                # Store the predicted change
                pred_mean_all[h, :] = pred_mean_h

                # Append the new prediction to update the history for the next iteration
                Y_fcst_start = np.vstack([Y_fcst_start[1:, :], pred_mean_h])

                # (Optional but good practice) Variance calculation for the current step
                x_t_kron = kron(np.eye(M), x_t.reshape(1, -1))
                pred_var_h = x_t_kron @ alpha_var @ x_t_kron.T + SIGMA_post
                pred_var_all[h, :, :] = pred_var_h

        pred_mean = pred_mean_all
        pred_var = pred_var_all
    else:
        pred_mean = None
        pred_var = None

    # Calculate number of parameters
    sigma_params = M * (M + 1) / 2

    if prior_type == 1:
        num_params = K * M + sigma_params
    elif prior_type == 2:
        num_params = K * M + sigma_params
    elif prior_type == 3:
        num_params = K * M + sigma_params

    else:
        raise ValueError("Unknown prior type.")

    try:
        Sigma_ML = SSE / T
        # numerical symmetry hygiene
        Sigma_ML = 0.5 * (Sigma_ML + Sigma_ML.T)

        # stable log|Σ| via slogdet, and check SPD by sign>0
        sign, logdet = np.linalg.slogdet(Sigma_ML)
        if sign <= 0:
            raise np.linalg.LinAlgError("Sigma_ML is not SPD.")

        # exact ML log-likelihood at Σ_ML (Gaussian)
        #  ℓ = -T/2 * [ M*log(2π) + log|Σ_ML| + M ]
        log_likelihood = -0.5 * T * (M * np.log(2 * np.pi) + logdet + M)

    except Exception as e:
        print(f"Log-likelihood computation failed at lag {p_lags}: {e}")
        log_likelihood = -np.inf

    return {
        "pred_mean": pred_mean,
        "pred_var": pred_var,
        "alpha_mean": alpha_mean,
        "alpha_var": alpha_var,
        "A_post": A_post,
        "SIGMA_post": SIGMA_post,
        "SIGMA_OLS": SIGMA_OLS,
        "log_likelihood": log_likelihood,
        "num_params": num_params,
        "SIGMA_samples": SIGMA_samples,
    }
