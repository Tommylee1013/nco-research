import numpy as np
import pandas as pd
from numpy.linalg import inv
from ..optimization.utils import symmetrize, nearest_positive_semidefinite

def black_litterman_posterior(
    prior_expected_returns: pd.Series,
    prior_covariance: pd.DataFrame,
    view_matrix: pd.DataFrame,
    view_values: pd.Series,
    view_uncertainty: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Black–Litterman posterior for linear views:
        view_matrix @ mu ~ N(view_values, view_uncertainty)
    Returns posterior (expected_returns, covariance).
    """
    mu = prior_expected_returns.values.reshape(-1,)
    sigma = symmetrize(prior_covariance).values
    P = view_matrix.values
    q = view_values.values
    omega = view_uncertainty.values

    PS = P @ sigma
    middle = inv(PS @ P.T + omega)
    mu_post = mu + (sigma @ P.T) @ (middle @ (q - P @ mu))
    sigma_post = sigma - (sigma @ P.T) @ (middle @ (P @ sigma))
    sigma_post = nearest_positive_semidefinite(pd.DataFrame(sigma_post, index=prior_covariance.index, columns=prior_covariance.columns)).values

    mu_post = pd.Series(mu_post, index=prior_expected_returns.index, name="expected_return")
    sigma_post = pd.DataFrame(sigma_post, index=prior_covariance.index, columns=prior_covariance.columns)
    return mu_post, sigma_post
