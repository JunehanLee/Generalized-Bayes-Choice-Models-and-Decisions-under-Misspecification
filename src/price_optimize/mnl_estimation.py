import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

def fit_hier_bayes_mnl(
    df: pd.DataFrame,
    J: int,
    *,
    id_col: str = "customer",
    y_col: str = "choice",
    price_prefix: str = "price_",   # price_1..price_J
    seed: int = 0,
    # priors (조정 가능)
    mu_beta_mean: float = 0,
    mu_beta_sd: float = 2.0,
    sigma_beta_rate: float = 1.0,
    mu_alpha_mean: float = 0,
    mu_alpha_sd: float = 2.0,
    sigma_alpha_rate: float = 1.0,
    alpha_lower: float = 0.0,
    **sample_kwargs
):
    """
    Hierarchical Bayesian MNL with outside baseline (0).

    Model (estimation; misspecified vs Normal-error DGP):
      - Customer-specific brand preference/sensitivity: beta[i,j]  for j=1..J
      - Customer-specific price sensitivity: alpha[i] (positive)
      - Utility:
          V_it0 = 0
          V_itj = beta[i,j] - alpha[i] * price_jt   (j=1..J)
      - Choice probs:
          pi_it = softmax([0, V_it1..V_itJ])
      - Likelihood:
          y_it ~ Categorical(pi_it), where y in {0..J}

    Returns
    -------
    trace : arviz.InferenceData
    meta : dict (useful for prediction)
    """
    df = df.sort_values([id_col, "period"] if "period" in df.columns else [id_col]).reset_index(drop=True)

    # prices: (Nobs, J) from price_1..price_J
    price_cols = [f"{price_prefix}{j}" for j in range(1, J + 1)]
    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}")

    P = df[price_cols].to_numpy(dtype=float)                # (Nobs, J)
    y = df[y_col].to_numpy().astype("int32")                # (Nobs,)
    if y.min() < 0 or y.max() > J:
        raise ValueError(f"{y_col} must be in [0..J]. Got min={y.min()}, max={y.max()}")

    # customer ids -> 0..I-1
    cust_id, cust_categories = pd.factorize(df[id_col], sort=True)
    cust_id = cust_id.astype("int64")
    I = int(cust_id.max() + 1)
    n_obs = len(df)

    with pm.Model() as model:
        # -------------------------
        # Hyperpriors for beta (brand preference)
        # -------------------------
        # 제품(브랜드)별 population mean/scale (j=1..J)
        mu_beta    = pm.Normal("mu_beta", mu_beta_mean, mu_beta_sd, shape=J)
        sigma_beta = pm.Exponential("sigma_beta", sigma_beta_rate, shape=J)

        # 고객별 beta[i,j]
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=(I, J))

        # -------------------------
        # Hyperpriors for alpha (price sensitivity)
        # -------------------------
        mu_alpha    = pm.Normal("mu_alpha", mu_alpha_mean, mu_alpha_sd)
        sigma_alpha = pm.Exponential("sigma_alpha", sigma_alpha_rate)

        # 고객별 alpha_i (양수 제약)
        alpha = pm.TruncatedNormal("alpha", mu=mu_alpha, sigma=sigma_alpha, lower=alpha_lower, shape=I)

        # -------------------------
        # Utilities and probs
        # -------------------------
        beta_obs  = beta[cust_id, :]            # (Nobs, J)
        alpha_obs = alpha[cust_id][:, None]     # (Nobs, 1)

        V_prod = beta_obs - alpha_obs * P       # (Nobs, J)

        # outside baseline=0을 앞에 붙여 (Nobs, J+1)
        V_all = pt.concatenate([pt.zeros((n_obs, 1)), V_prod], axis=1)

        # softmax -> probabilities
        pi = pm.math.softmax(V_all, axis=1)     # (Nobs, J+1)

        # likelihood
        pm.Categorical("y", p=pi, observed=y)

        trace = pm.sample(random_seed=seed, **sample_kwargs)

    meta = {
        "I": I,
        "J": J,
        "price_cols": price_cols,
        "id_col": id_col,
        "y_col": y_col,
        "categories": cust_categories,
    }
    return trace, meta


def predict_hier_bayes_mnl(df: pd.DataFrame, trace, meta: dict) -> dict:
    """
    Posterior mean probs and predicted class for each row in df.
    df must contain the same columns used in training.
    """
    id_col = meta["id_col"]
    y_col = meta["y_col"]
    price_cols = meta["price_cols"]
    J = meta["J"]

    df = df.sort_values([id_col, "period"] if "period" in df.columns else [id_col]).reset_index(drop=True)

    P = df[price_cols].to_numpy(dtype=float)  # (Nobs, J)
    cust_id, cust_categories = pd.factorize(df[id_col], sort=True)
    cust_id = cust_id.astype("int64")
    n_obs = len(df)

    # stack posterior draws
    beta_xr = trace.posterior["beta"].stack(draws=("chain", "draw"))
    beta_xr = beta_xr.transpose("beta_dim_0", "beta_dim_1", "draws")
    beta = beta_xr.values  # (I, J, D)

    alpha_xr = trace.posterior["alpha"].stack(draws=("chain", "draw"))
    alpha_xr = alpha_xr.transpose("alpha_dim_0", "draws")
    alpha = alpha_xr.values  # (I, D)

    I, J2, D = beta.shape
    assert J2 == J

    beta_obs = beta[cust_id, :, :]                 # (Nobs, J, D)
    alpha_obs = alpha[cust_id, :][:, None, :]      # (Nobs, 1, D)

    V_prod = beta_obs - alpha_obs * P[:, :, None]  # (Nobs, J, D)
    V0 = np.zeros((n_obs, 1, D))                   # outside baseline
    V_all = np.concatenate([V0, V_prod], axis=1)   # (Nobs, J+1, D)

    # softmax over axis=1
    Vmax = V_all.max(axis=1, keepdims=True)
    expV = np.exp(V_all - Vmax)
    probs = expV / expV.sum(axis=1, keepdims=True)  # (Nobs, J+1, D)

    p_mean = probs.mean(axis=2)  # (Nobs, J+1)
    y_pred = p_mean.argmax(axis=1).astype(np.int64)

    return {
        "p_mean": p_mean,          # (Nobs, J+1)
        "y_pred": y_pred,          # (Nobs,)
        "alpha_draws": alpha,      # (I, D)
        "beta_draws": beta,        # (I, J, D)
        "meta": {"I": I, "J": J, "D": D, "categories": cust_categories},
    }

def mnl_loss_fn(y_true, pi_pred, kind="ce", alpha=None):
    """
    Loss for multinomial choice model.

    Args:
        y_true: shape (N,), integer labels in {0, ..., J}
        pi_pred: shape (N, J+1), predicted class probabilities
        kind:
            - "ce": multiclass cross-entropy
            - "squared": squared error against one-hot target
            - "huber": huber loss against one-hot target
            - "sph": scaled pseudo-huber against one-hot target
    """
    pi_clipped = pt.clip(pi_pred, 1e-8, 1 - 1e-8)
    n_obs = y_true.shape[0]
    n_class = pi_pred.shape[1]

    # one-hot target matrix: (N, J+1)
    Y = pt.extra_ops.to_one_hot(y_true, n_class)

    if kind == "ce":
        return -pt.sum(pt.log(pi_clipped)[pt.arange(n_obs), y_true])

    elif kind == "squared":
        return pt.sum(pt.sqr(Y - pi_clipped))

    elif kind == "huber":
        delta = 1.0
        residual = Y - pi_clipped
        return pt.sum(
            pt.switch(
                pt.abs(residual) <= delta,
                0.5 * residual**2,
                delta * (pt.abs(residual) - 0.5 * delta)
            )
        )

    elif kind == "sph":
        if alpha is None:
            raise ValueError("alpha is required for SPH loss")
        t = Y - pi_clipped
        scale = alpha * pt.sqrt(1.0 + alpha**2)
        loss_ij = scale * (pt.sqrt(1.0 + (t / alpha)**2) - 1.0)
        return pt.sum(loss_ij)

    else:
        raise ValueError("Unknown loss kind")


def fit_hier_generalized_bayes_mnl(
    df,
    J,
    *,
    id_col="customer",
    y_col="choice",
    price_prefix: str = "price_",   # price_1..price_J
    mu_beta_mean=1.0,
    mu_beta_sd=1.0,
    sigma_beta_rate=2.0,
    mu_alpha_mean=0.5,
    mu_alpha_sd=0.5,
    sigma_alpha_rate=2.0,
    alpha_lower=0.0,
    lam=1.0,
    loss_kind="ce",
    estimate_sph_alpha=True,
    seed=42,
    **sample_kwargs
):
    """
    Generalized Bayes hierarchical multinomial choice model.
    Same hierarchical structure as the standard Bayes baseline,
    but replaces likelihood with exp(-lam * loss).
    """
    df = df.sort_values([id_col, "period"] if "period" in df.columns else [id_col]).reset_index(drop=True)

    # prices: (Nobs, J) from price_1..price_J
    price_cols = [f"{price_prefix}{j}" for j in range(1, J + 1)]
    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}")

    P = df[price_cols].to_numpy(dtype=float)                # (Nobs, J)
    y = df[y_col].to_numpy().astype("int32")                # (Nobs,)
    if y.min() < 0 or y.max() > J:
        raise ValueError(f"{y_col} must be in [0..J]. Got min={y.min()}, max={y.max()}")

    # customer ids -> 0..I-1
    cust_id, cust_categories = pd.factorize(df[id_col], sort=True)
    cust_id = cust_id.astype("int64")
    I = int(cust_id.max() + 1)
    n_obs = len(df)

    with pm.Model() as model:
        # -------------------------
        # Hyperpriors for beta (brand preference)
        # -------------------------
        # 제품(브랜드)별 population mean/scale (j=1..J)
        mu_beta    = pm.Normal("mu_beta", mu_beta_mean, mu_beta_sd, shape=J)
        sigma_beta = pm.Exponential("sigma_beta", sigma_beta_rate, shape=J)

        # 고객별 beta[i,j]
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=(I, J))

        # -------------------------
        # Hyperpriors for alpha (price sensitivity)
        # -------------------------
        mu_alpha    = pm.Normal("mu_alpha", mu_alpha_mean, mu_alpha_sd)
        sigma_alpha = pm.Exponential("sigma_alpha", sigma_alpha_rate)

        # 고객별 alpha_i (양수 제약)
        alpha = pm.TruncatedNormal("alpha", mu=mu_alpha, sigma=sigma_alpha, lower=alpha_lower, shape=I)

        # -------------------------
        # Utilities and probs
        # -------------------------
        beta_obs  = beta[cust_id, :]            # (Nobs, J)
        alpha_obs = alpha[cust_id][:, None]     # (Nobs, 1)

        V_prod = beta_obs - alpha_obs * P       # (Nobs, J)

        # outside baseline=0을 앞에 붙여 (Nobs, J+1)
        V_all = pt.concatenate([pt.zeros((n_obs, 1)), V_prod], axis=1)

        # softmax -> probabilities
        pi = pm.math.softmax(V_all, axis=1)     # (Nobs, J+1)
        # -------------------------
        # Generalized Bayes loss
        # posterior ∝ prior * exp(-lam * loss)
        # -------------------------
        if loss_kind == "sph":
            if estimate_sph_alpha:
                alpha_sq_loss = pm.Gamma("alpha_sq_loss", alpha=1.0, beta=1.0)
                alpha_loss = pm.Deterministic("alpha_loss", pt.sqrt(alpha_sq_loss))
                loss = mnl_loss_fn(y, pi, kind="sph", alpha=alpha_loss)
            else:
                alpha_loss = 1.0
                loss = mnl_loss_fn(y, pi, kind="sph", alpha=alpha_loss)
        else:
            loss = mnl_loss_fn(y, pi, kind=loss_kind)

        pm.Potential(f"{loss_kind}_loss", -lam * loss)

        trace = pm.sample(random_seed=seed, **sample_kwargs)
    meta = {
        "I": I,
        "J": J,
        "price_cols": price_cols,
        "id_col": id_col,
        "y_col": y_col,
        "categories": cust_categories,
    }
    return trace, meta
