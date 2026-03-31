from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd


# =========================================================
# 1. Result containers
# =========================================================

@dataclass
class PeriodOptimizationResult:
    period: int
    observed_price_vec: np.ndarray
    optimal_price_A: float
    optimal_revenue_A: float
    revenue_curve: np.ndarray


@dataclass
class PricingExperimentResult:
    observed_revenue_A: float
    counterfactual_revenue_A_mean: float
    counterfactual_revenue_A_std: float
    counterfactual_revenue_A_values: np.ndarray
    optimal_price_path_A: np.ndarray
    observed_price_path_A: np.ndarray
    observed_price_path_B: np.ndarray
    observed_price_path_C: np.ndarray
    period_results: List[PeriodOptimizationResult]


# =========================================================
# 2. MNL probability helpers
# =========================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def compute_choice_probs_mnl(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    price_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute MNL choice probabilities with outside option utility fixed at 0.

    Parameters
    ----------
    alpha_arr : np.ndarray, shape (N,)
        Customer-specific price sensitivities.
    beta_mat : np.ndarray, shape (N, J)
        Customer-specific product preferences.
    price_vec : np.ndarray, shape (J,)
        Product prices for one period.

    Returns
    -------
    prob_mat : np.ndarray, shape (N, J+1)
        Column 0 is outside option.
        Columns 1..J are product choice probabilities.
    """
    alpha_arr = np.asarray(alpha_arr, dtype=float)
    beta_mat = np.asarray(beta_mat, dtype=float)
    price_vec = np.asarray(price_vec, dtype=float)

    if alpha_arr.ndim != 1:
        raise ValueError("alpha_arr must be 1D.")
    if beta_mat.ndim != 2:
        raise ValueError("beta_mat must be 2D.")
    if beta_mat.shape[0] != alpha_arr.shape[0]:
        raise ValueError("alpha_arr and beta_mat must have matching first dimension.")
    if beta_mat.shape[1] != price_vec.shape[0]:
        raise ValueError("price_vec length must equal number of products.")

    util_products = beta_mat - alpha_arr[:, None] * price_vec[None, :]
    util_full = np.concatenate(
        [np.zeros((alpha_arr.shape[0], 1)), util_products],
        axis=1,
    )
    return softmax(util_full, axis=1)


# =========================================================
# 3. Single-period expected revenue for A
# =========================================================

def compute_expected_revenue_A(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    price_vec: np.ndarray,
    target_product_idx: int = 1,
) -> float:
    """
    Compute expected revenue for product A (or target product) under MNL.

    Since price is treated as net-of-cost value:
        revenue_A = price_A * expected_count_A
    """
    prob_mat = compute_choice_probs_mnl(
        alpha_arr=alpha_arr,
        beta_mat=beta_mat,
        price_vec=price_vec,
    )

    j = target_product_idx
    if j < 1 or j > price_vec.shape[0]:
        raise ValueError("target_product_idx must be in {1, ..., J}.")

    expected_count_A = prob_mat[:, j].sum()
    price_A = price_vec[j - 1]

    return float(price_A * expected_count_A)


def optimize_price_A_t(
    alpha_arr: np.ndarray,
    beta_mat: np.ndarray,
    observed_price_vec_t: np.ndarray,
    price_grid: np.ndarray,
    target_product_idx: int = 1,
    period: Optional[int] = None,
) -> PeriodOptimizationResult:
    """
    Optimize A's price for one period t while keeping other product prices fixed
    at their observed period-t values.
    """
    observed_price_vec_t = np.asarray(observed_price_vec_t, dtype=float).copy()
    price_grid = np.asarray(price_grid, dtype=float)

    values = []

    for p in price_grid:
        candidate_price_vec = observed_price_vec_t.copy()
        candidate_price_vec[target_product_idx - 1] = p

        revenue_A = compute_expected_revenue_A(
            alpha_arr=alpha_arr,
            beta_mat=beta_mat,
            price_vec=candidate_price_vec,
            target_product_idx=target_product_idx,
        )
        values.append(revenue_A)

    values_arr = np.asarray(values, dtype=float)
    best_idx = int(np.argmax(values_arr))

    return PeriodOptimizationResult(
        period=-1 if period is None else int(period),
        observed_price_vec=observed_price_vec_t,
        optimal_price_A=float(price_grid[best_idx]),
        optimal_revenue_A=float(values_arr[best_idx]),
        revenue_curve=values_arr,
    )


# =========================================================
# 4. Observed price path extraction
# =========================================================

def extract_price_path(
    df: pd.DataFrame,
    n_periods: int,
    n_products: int,
    period_col: str = "period",
    price_col_prefix: str = "price_",
) -> np.ndarray:
    """
    Extract observed price path of shape (J, T) from observed dataframe.

    Assumes prices are common across customers within each period.
    """
    if period_col not in df.columns:
        raise ValueError(f"df must contain '{period_col}' column.")

    price_path = np.zeros((n_products, n_periods), dtype=float)

    for t in range(n_periods):
        df_t = df[df[period_col] == t]
        if df_t.empty:
            raise ValueError(f"No rows found for period {t}.")

        for j in range(1, n_products + 1):
            col = f"{price_col_prefix}{j}"
            if col not in df.columns:
                raise ValueError(f"Missing price column: {col}")

            unique_prices = df_t[col].unique()
            if len(unique_prices) != 1:
                raise ValueError(
                    f"Price column {col} has multiple values in period {t}. "
                    "Expected common posted price within each period."
                )

            price_path[j - 1, t] = float(unique_prices[0])

    return price_path


# =========================================================
# 5. Observed revenue from observed dataset
# =========================================================

def compute_observed_revenue_A(
    df: pd.DataFrame,
    target_product_idx: int = 1,
    price_col_prefix: str = "price_",
) -> float:
    """
    Compute observed revenue for product A from observed dataset:
        sum(price_A over rows where choice == A)
    """
    choice_col = "choice"
    price_col = f"{price_col_prefix}{target_product_idx}"

    if choice_col not in df.columns:
        raise ValueError("df must contain 'choice'.")
    if price_col not in df.columns:
        raise ValueError(f"df must contain '{price_col}'.")

    chosen_mask = df[choice_col].to_numpy() == target_product_idx
    return float(df.loc[chosen_mask, price_col].to_numpy(dtype=float).sum())


# =========================================================
# 6. Period-by-period price path optimization
# =========================================================

def optimize_price_path_A(
    observed_df: pd.DataFrame,
    est_alpha_arr: np.ndarray,
    est_beta_mat: np.ndarray,
    price_grid: np.ndarray,
    n_periods: int,
    n_products: int = 3,
    target_product_idx: int = 1,
    period_col: str = "period",
    price_col_prefix: str = "price_",
) -> List[PeriodOptimizationResult]:
    """
    For each period t:
    - use observed period-t price vector
    - keep B,C fixed at their observed period-t values
    - optimize A only
    """
    observed_price_path = extract_price_path(
        df=observed_df,
        n_periods=n_periods,
        n_products=n_products,
        period_col=period_col,
        price_col_prefix=price_col_prefix,
    )

    results: List[PeriodOptimizationResult] = []

    for t in range(n_periods):
        observed_price_vec_t = observed_price_path[:, t]

        result_t = optimize_price_A_t(
            alpha_arr=est_alpha_arr,
            beta_mat=est_beta_mat,
            observed_price_vec_t=observed_price_vec_t,
            price_grid=price_grid,
            target_product_idx=target_product_idx,
            period=t,
        )
        results.append(result_t)

    return results


# =========================================================
# 7. Counterfactual price schedule construction
# =========================================================

def build_cf_price_schedule(
    period_results: List[PeriodOptimizationResult],
    target_product_idx: int = 1,
) -> np.ndarray:
    """
    Build counterfactual price schedule of shape (J, T):
    - A uses optimal period-specific prices
    - B,C remain at observed period-specific prices
    """
    if len(period_results) == 0:
        raise ValueError("period_results is empty.")

    n_periods = len(period_results)
    n_products = len(period_results[0].observed_price_vec)

    price_schedule_cf = np.zeros((n_products, n_periods), dtype=float)

    for t, res in enumerate(period_results):
        price_vec_t = res.observed_price_vec.copy()
        price_vec_t[target_product_idx - 1] = res.optimal_price_A
        price_schedule_cf[:, t] = price_vec_t

    return price_schedule_cf


# =========================================================
# 8. Counterfactual revenue evaluation
# =========================================================

def compute_cf_revenue_A(
    df_cf: pd.DataFrame,
    target_product_idx: int = 1,
    price_col_prefix: str = "price_",
) -> float:
    """
    Compute counterfactual revenue for product A from generated dataframe:
        sum(price_A over rows where choice == A)
    """
    choice_col = "choice"
    price_col = f"{price_col_prefix}{target_product_idx}"

    if choice_col not in df_cf.columns:
        raise ValueError("df_cf must contain 'choice'.")
    if price_col not in df_cf.columns:
        raise ValueError(f"df_cf must contain '{price_col}'.")

    chosen_mask = df_cf[choice_col].to_numpy() == target_product_idx
    return float(df_cf.loc[chosen_mask, price_col].to_numpy(dtype=float).sum())


def simulate_cf_revenue(
    generate_multinomial_dgp_func,
    true_alpha_arr: np.ndarray,
    true_beta_mat: np.ndarray,
    price_schedule_cf: np.ndarray,
    target_product_idx: int = 1,
    n_rep: int = 100,
    true_error_type: str = "probit",
    base_seed: Optional[int] = None,
    price_col_prefix: str = "price_",
    **dgp_kwargs: Any,
) -> np.ndarray:
    """
    Simulate counterfactual revenue under the true DGP using:
    - optimal A price path
    - observed B,C price paths
    """
    true_alpha_arr = np.asarray(true_alpha_arr, dtype=float)
    true_beta_mat = np.asarray(true_beta_mat, dtype=float)

    n_customers = true_alpha_arr.shape[0]
    n_products, n_periods = price_schedule_cf.shape

    values = []

    for r in range(n_rep):
        seed_r = None if base_seed is None else base_seed + r

        df_cf, _ = generate_multinomial_dgp_func(
            n_customers=n_customers,
            n_products=n_products,
            n_periods=n_periods,
            beta_mat=true_beta_mat,
            alpha_arr=true_alpha_arr,
            price_schedule=price_schedule_cf,
            seed=seed_r,
            error_type=true_error_type,
            **dgp_kwargs,
        )

        revenue_r = compute_cf_revenue_A(
            df_cf=df_cf,
            target_product_idx=target_product_idx,
            price_col_prefix=price_col_prefix,
        )
        values.append(revenue_r)

    return np.asarray(values, dtype=float)


# =========================================================
# 9. Full pricing experiment
# =========================================================

def run_pricing_experiment(
    observed_df: pd.DataFrame,
    generate_multinomial_dgp_func,
    true_alpha_arr: np.ndarray,
    true_beta_mat: np.ndarray,
    est_alpha_arr: np.ndarray,
    est_beta_mat: np.ndarray,
    price_grid: np.ndarray,
    n_periods: int,
    n_products: int = 3,
    target_product_idx: int = 1,
    n_counterfactual_rep: int = 100,
    true_error_type: str = "probit",
    base_seed: Optional[int] = None,
    period_col: str = "period",
    price_col_prefix: str = "price_",
    **dgp_kwargs: Any,
) -> PricingExperimentResult:
    """
    Full pipeline:

    1) compute observed revenue for A from observed data
    2) optimize period-specific A price path using fitted MNL
    3) build counterfactual price schedule
    4) simulate counterfactual revenue under the true DGP
    """
    observed_revenue_A = compute_observed_revenue_A(
        df=observed_df,
        target_product_idx=target_product_idx,
        price_col_prefix=price_col_prefix,
    )

    period_results = optimize_price_path_A(
        observed_df=observed_df,
        est_alpha_arr=est_alpha_arr,
        est_beta_mat=est_beta_mat,
        price_grid=price_grid,
        n_periods=n_periods,
        n_products=n_products,
        target_product_idx=target_product_idx,
        period_col=period_col,
        price_col_prefix=price_col_prefix,
    )

    price_schedule_cf = build_cf_price_schedule(
        period_results=period_results,
        target_product_idx=target_product_idx,
    )

    cf_values = simulate_cf_revenue(
        generate_multinomial_dgp_func=generate_multinomial_dgp_func,
        true_alpha_arr=true_alpha_arr,
        true_beta_mat=true_beta_mat,
        price_schedule_cf=price_schedule_cf,
        target_product_idx=target_product_idx,
        n_rep=n_counterfactual_rep,
        true_error_type=true_error_type,
        base_seed=base_seed,
        price_col_prefix=price_col_prefix,
        **dgp_kwargs,
    )

    observed_price_path = extract_price_path(
        df=observed_df,
        n_periods=n_periods,
        n_products=n_products,
        period_col=period_col,
        price_col_prefix=price_col_prefix,
    )

    optimal_price_path_A = np.array(
        [res.optimal_price_A for res in period_results],
        dtype=float,
    )

    return PricingExperimentResult(
        observed_revenue_A=float(observed_revenue_A),
        counterfactual_revenue_A_mean=float(cf_values.mean()),
        counterfactual_revenue_A_std=float(cf_values.std(ddof=1)) if len(cf_values) > 1 else 0.0,
        counterfactual_revenue_A_values=cf_values,
        optimal_price_path_A=optimal_price_path_A,
        observed_price_path_A=observed_price_path[0, :].copy(),
        observed_price_path_B=observed_price_path[1, :].copy(),
        observed_price_path_C=observed_price_path[2, :].copy(),
        period_results=period_results,
    )
