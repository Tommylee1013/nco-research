import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from .utils import nearest_correlation, covariance_to_correlation, correlation_distance, nearest_positive_semidefinite

def _recursive_allocation(node, asset_names, expected_returns, covariance_matrix) -> pd.Series:
    """
    Recursively allocate weights in the NCO tree using two-branch minimum-variance at each split.
    """
    if node.is_leaf():
        weights = pd.Series(0.0, index=asset_names)
        weights.iloc[node.id] = 1.0
        return weights

    left = _recursive_allocation(node.left, asset_names, expected_returns, covariance_matrix)
    right = _recursive_allocation(node.right, asset_names, expected_returns, covariance_matrix)

    left_idx = left[left > 0].index
    right_idx = right[right > 0].index
    left.loc[left_idx] /= left.loc[left_idx].sum()
    right.loc[right_idx] /= right.loc[right_idx].sum()

    var_left = float(left @ (covariance_matrix @ left))
    var_right = float(right @ (covariance_matrix @ right))
    cov_lr = float(left @ (covariance_matrix @ right))
    meta_cov = pd.DataFrame([[var_left, cov_lr], [cov_lr, var_right]], index=["L","R"], columns=["L","R"])
    meta_cov = nearest_positive_semidefinite(meta_cov)
    # meta_cov (2x2 covariance) computed but unused in current version.
    # Could be useful if extending to Sharpe-based or generalized splits.

    if (var_left + var_right) <= 1e-12:
        alpha_left, alpha_right = 0.5, 0.5
    else:
        alpha_left, alpha_right = var_right / (var_left + var_right), var_left / (var_left + var_right)

    return alpha_left * left + alpha_right * right

def nco_portfolio(expected_returns: pd.Series, covariance_matrix: pd.DataFrame, correlation_matrix: pd.DataFrame | None = None, method: str = "single") -> pd.Series:
    """
    Compute NCO weights:
      1) Build tree from correlation distance.
      2) Recursively allocate min-variance between siblings.
      3) Return long-only weights that sum to 1.
    """
    if correlation_matrix is None:
        correlation_matrix = covariance_to_correlation(covariance_matrix)
    correlation_matrix = nearest_correlation(correlation_matrix)

    distance = correlation_distance(correlation_matrix)
    condensed = squareform(distance.values, checks=False)
    link = linkage(condensed, method=method)
    root, _ = to_tree(link, rd=True)

    weights = _recursive_allocation(root, covariance_matrix.index, expected_returns, covariance_matrix)
    return weights / weights.sum()
