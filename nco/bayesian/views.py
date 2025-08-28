import numpy as np
import pandas as pd

def build_pairwise_relative_views(
    asset_names: list[str],
    base_expected_returns: pd.Series,
    n_views: int = 10,
    noise_std: float = 1e-4,
    mis_spec_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Create relative (pairwise) views of the form: e_i - e_j applied to μ.
    q is set near (μ_i - μ_j) with optional noise and mis-specification (sign flip).
    Args:
        base_expected_returns: typically the in-sample estimator (no leakage).
        mis_spec_probability: probability to flip the sign of the true difference.
    Returns:
        view_matrix P (k x N), view_values q (k,), view_uncertainty Ω (k x k) [diagonal; to be filled by calibrator]
    """
    if rng is None:
        rng = np.random.default_rng()
    n_assets = len(asset_names)
    rows = []
    q_values = []
    for _ in range(n_views):
        i, j = rng.integers(0, n_assets, size=2)
        if i == j:
            j = (j + 1) % n_assets
        row = np.zeros(n_assets)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)

        true_diff = float(base_expected_returns.iloc[i] - base_expected_returns.iloc[j])
        if rng.random() < mis_spec_probability:
            true_diff = -true_diff
        q_values.append(true_diff + rng.normal(0.0, noise_std))

    P = pd.DataFrame(rows, columns=asset_names)
    q = pd.Series(q_values, index=P.index, name="view_value")
    # Ω는 calibrate_view_uncertainty()에서 설정
    Omega = pd.DataFrame(np.eye(n_views), index=P.index, columns=P.index)  # placeholder
    return P, q, Omega


def build_absolute_views(
    asset_names: list[str],
    base_expected_returns: pd.Series,
    n_views: int = 10,
    noise_std: float = 1e-4,
    mis_spec_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Create absolute views of the form: e_i^T μ ≈ q_i.
    q is near μ_i with optional noise and mis-specification (sign flip around 0 baseline).
    """
    if rng is None:
        rng = np.random.default_rng()
    n_assets = len(asset_names)
    chosen = rng.choice(n_assets, size=n_views, replace=False if n_views <= n_assets else True)
    rows = []
    q_values = []
    for idx in chosen:
        row = np.zeros(n_assets)
        row[idx] = 1.0
        rows.append(row)

        val = float(base_expected_returns.iloc[idx])
        if rng.random() < mis_spec_probability:
            val = -val
        q_values.append(val + rng.normal(0.0, noise_std))

    P = pd.DataFrame(rows, columns=asset_names)
    q = pd.Series(q_values, index=P.index, name="view_value")
    Omega = pd.DataFrame(np.eye(len(q_values)), index=P.index, columns=P.index)
    return P, q, Omega


def build_cluster_relative_views(
    asset_names: list[str],
    n_clusters: int,
    base_expected_returns: pd.Series,
    n_views: int = 5,
    noise_std: float = 1e-4,
    mis_spec_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[list[str]]]:
    """
    Create cluster-vs-cluster relative views:
      (avg μ over cluster A) - (avg μ over cluster B) ≈ q
    Cluster partition follows equal split over asset order (same as generator).
    Returns clusters for logging/analysis.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_assets = len(asset_names)
    cluster_indices = np.array_split(np.arange(n_assets), n_clusters)
    clusters = [[asset_names[i] for i in idx] for idx in cluster_indices]

    rows, q_values = [], []
    for _ in range(n_views):
        a, b = rng.integers(0, n_clusters, size=2)
        if a == b:
            b = (b + 1) % n_clusters
        row = np.zeros(n_assets)
        row[[asset_names.index(n) for n in clusters[a]]] = 1.0 / len(clusters[a])
        row[[asset_names.index(n) for n in clusters[b]]] = -1.0 / len(clusters[b])
        rows.append(row)

        true_diff = float(base_expected_returns.loc[clusters[a]].mean() - base_expected_returns.loc[clusters[b]].mean())
        if rng.random() < mis_spec_probability:
            true_diff = -true_diff
        q_values.append(true_diff + rng.normal(0.0, noise_std))

    P = pd.DataFrame(rows, columns=asset_names)
    q = pd.Series(q_values, index=P.index, name="view_value")
    Omega = pd.DataFrame(np.eye(len(q_values)), index=P.index, columns=P.index)
    return P, q, Omega, clusters


def calibrate_view_uncertainty(
    view_matrix: pd.DataFrame,
    prior_covariance: pd.DataFrame,
    scale: float = 1.0,
) -> pd.DataFrame:
    """
    Set Ω diagonally proportional to diag(P Σ P^T):
        Ω = scale * diag(diag(P Σ P^T))
    This ties confidence to the variance of each view-implied linear form.
    """
    proj_var = (view_matrix.values @ prior_covariance.values @ view_matrix.values.T)
    omega_diag = np.diag(np.clip(np.diag(proj_var), 1e-16, None)) * scale
    return pd.DataFrame(omega_diag, index=view_matrix.index, columns=view_matrix.index)
