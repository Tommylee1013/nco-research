import numpy as np
import pandas as pd
from .utils import nearest_correlation

def make_correlation_view_from_true(
    true_correlation: pd.DataFrame,
    intra_scale: float = 1.0,
    inter_scale: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Create a correlation 'view' by scaling intra-cluster and inter-cluster correlations
    and adding small symmetric noise, then projecting to nearest correlation matrix.
    Assumes assets are ordered by equal-split clusters (as in generator).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = true_correlation.shape[0]
    n_clusters_guess = max(2, int(np.round(np.sqrt(n)/1.5)))  # heuristic if not provided
    cluster_indices = np.array_split(np.arange(n), n_clusters_guess)

    R = true_correlation.values.copy()
    # scale intra vs inter roughly
    mask = np.zeros_like(R, dtype=bool)
    for idx in cluster_indices:
        mask[np.ix_(idx, idx)] = True
    R_intra = R.copy(); R_inter = R.copy()
    R_intra[~mask] = 0.0
    R_inter[mask] = 0.0
    R_view = intra_scale * R_intra + inter_scale * R_inter

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=R.shape)
        noise = 0.5 * (noise + noise.T)
        np.fill_diagonal(noise, 0.0)
        R_view = R_view + noise

    np.fill_diagonal(R_view, 1.0)
    R_view = np.clip(R_view, -0.99, 1.0)
    return nearest_correlation(pd.DataFrame(R_view, index=true_correlation.index, columns=true_correlation.columns))
