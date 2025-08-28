import numpy as np
import pandas as pd
from numpy.linalg import inv

def minimum_variance_portfolio(covariance_matrix: pd.DataFrame) -> pd.Series:
    """
    Long-only minimum variance portfolio via active-set pruning.
    """
    assets = covariance_matrix.index
    active_assets = assets
    weights = pd.Series(0.0, index=assets)
    while True:
        sub_cov = covariance_matrix.loc[active_assets, active_assets].values
        inv_cov = inv(sub_cov)
        ones = np.ones(len(active_assets))
        sub_weights = (inv_cov @ ones) / (ones @ inv_cov @ ones)
        if np.all(sub_weights >= -1e-12):
            weights.loc[active_assets] = np.clip(sub_weights, 0, None)
            return weights / weights.sum()
        worst = active_assets[np.argmin(sub_weights)]
        active_assets = active_assets.drop(worst)

def maximum_sharpe_portfolio(expected_returns: pd.Series, covariance_matrix: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Long-only maximum Sharpe ratio portfolio via active-set pruning.
    """
    assets = covariance_matrix.index
    active_assets = assets
    weights = pd.Series(0.0, index=assets)

    while True:
        sub_cov = covariance_matrix.loc[active_assets, active_assets].values
        sub_ret = (expected_returns.loc[active_assets] - risk_free_rate).values
        inv_cov = inv(sub_cov)
        sub_weights = inv_cov @ sub_ret

        if np.all(sub_weights >= -1e-12):
            if sub_weights.sum() <= 0:
                # here
                fallback = minimum_variance_portfolio(covariance_matrix.loc[active_assets, active_assets])
                weights.loc[active_assets] = fallback.values
                return weights / weights.sum()
            weights.loc[active_assets] = np.clip(sub_weights, 0, None)
            return weights / weights.sum()
        else:
            worst = active_assets[np.argmin(sub_weights)]
            active_assets = active_assets.drop(worst)

def inverse_variance_portfolio(covariance_matrix: pd.DataFrame) -> pd.Series:
    """
    Inverse-variance portfolio (IVP).
    """
    inv_var = 1.0 / np.diag(covariance_matrix.values)
    weights = pd.Series(inv_var, index=covariance_matrix.index)
    return weights / weights.sum()