import numpy as np
import pandas as pd

def ledoit_wolf_shrinkage_identity(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit–Wolf (2004) shrinkage toward scaled identity (σ^2 I).
    Returns a shrunk covariance estimator.
    """
    X = returns.values
    t_len, n_assets = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    sample_cov = np.cov(Xc, rowvar=False, ddof=1)

    avg_var = np.trace(sample_cov) / n_assets
    target = avg_var * np.eye(n_assets)

    # simplified and stable estimator of phi (total variance of sample covariance elements)
    diff_sq = (Xc[:, :, None] * Xc[:, None, :])  # T x N x N outer products
    # unbiased covariance of outer products approximated by sample covariance; simplified phi:
    phi = np.sum((np.cov(Xc, rowvar=False, ddof=1) - sample_cov) ** 2)

    gamma = np.linalg.norm(sample_cov - target, ord="fro") ** 2
    delta = 0.0 if gamma == 0 else np.clip(phi / (t_len * gamma), 0.0, 1.0)

    shrunk = delta * target + (1.0 - delta) * sample_cov
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)

def ledoit_wolf_shrinkage_constant_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit–Wolf (2003) shrinkage toward constant-correlation target.
    Off-diagonal correlations share one average value; variances are preserved.
    """
    X = returns.values
    t_len, n_assets = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    sample_cov = np.cov(Xc, rowvar=False, ddof=1)

    std = np.sqrt(np.diag(sample_cov))
    inv_std = np.diag(1.0 / np.clip(std, 1e-12, None))
    sample_corr = inv_std @ sample_cov @ inv_std
    np.fill_diagonal(sample_corr, 1.0)

    r_bar = (sample_corr.sum() - n_assets) / (n_assets * (n_assets - 1))
    corr_target = np.full_like(sample_corr, r_bar)
    np.fill_diagonal(corr_target, 1.0)
    target = np.diag(std) @ corr_target @ np.diag(std)

    # simplified phi/gamma (stable in practice)
    phi = np.sum((np.cov(Xc, rowvar=False, ddof=1) - sample_cov) ** 2)
    gamma = np.linalg.norm(sample_cov - target, ord="fro") ** 2
    delta = 0.0 if gamma == 0 else np.clip(phi / (t_len * gamma), 0.0, 1.0)

    shrunk = delta * target + (1.0 - delta) * sample_cov
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
