import numpy as np
import pandas as pd
from ..optimization.utils import nearest_correlation, correlation_to_covariance

def generate_block_covariance(
    asset_names: list[str],
    n_clusters: int = 4,
    intra_cluster_correlation: float = 0.6,
    inter_cluster_correlation: float = 0.1,
    min_volatility: float = 0.15,
    max_volatility: float = 0.30,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a block-structured correlation and covariance matrix.
    """
    rng = np.random.default_rng(seed)
    n = len(asset_names)
    clusters = np.array_split(np.arange(n), n_clusters)

    correlation = pd.DataFrame(inter_cluster_correlation, index=asset_names, columns=asset_names)
    for idx in clusters:
        labels = [asset_names[i] for i in idx]
        correlation.loc[labels, labels] = intra_cluster_correlation
    np.fill_diagonal(correlation.values, 1.0)
    correlation = nearest_correlation(correlation)

    volatility = pd.Series(rng.uniform(min_volatility, max_volatility, size=n), index=asset_names, name="volatility")
    covariance = correlation_to_covariance(correlation, volatility)
    return correlation, covariance

def simulate_returns(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    n_observations: int,
    degrees_of_freedom: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate returns: Gaussian if degrees_of_freedom is None; Student-t otherwise.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(expected_returns)
    if degrees_of_freedom is None:
        samples = rng.multivariate_normal(mean=expected_returns.values, cov=covariance_matrix.values, size=n_observations)
        return pd.DataFrame(samples, columns=expected_returns.index)
    z = rng.multivariate_normal(mean=np.zeros(n_assets), cov=covariance_matrix.values, size=n_observations)
    w = rng.chisquare(degrees_of_freedom, size=n_observations) / degrees_of_freedom
    samples = expected_returns.values + z / np.sqrt(w)[:, None]
    return pd.DataFrame(samples, columns=expected_returns.index)
