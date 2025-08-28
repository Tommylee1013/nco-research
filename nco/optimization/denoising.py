import numpy as np
import pandas as pd
from numpy.linalg import eigh

def eigen_decomposition(correlation_matrix: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Eigen-decompose a correlation matrix and return eigenvalues (descending) and eigenvectors.
    """
    correlation_matrix = 0.5 * (correlation_matrix + correlation_matrix.T)
    values, vectors = eigh(correlation_matrix.values)   # ascending
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    eigenvalues = pd.Series(values, index=[f"λ{i+1}" for i in range(len(values))], name="eigenvalue")
    eigenvectors = pd.DataFrame(vectors, index=correlation_matrix.index, columns=eigenvalues.index)
    return eigenvalues, eigenvectors

def mp_bulk_edge(q: float, sigma2: float = 1.0) -> float:
    """
    Marčenko–Pastur bulk right edge (lambda_max) for aspect ratio q = T/N and variance sigma2.
    """
    if q <= 0:
        raise ValueError("q must be positive (q = T/N).")
    return float(sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2)

def count_signal_factors(eigenvalues: pd.Series, q: float, sigma2: float | None = None) -> int:
    """
    Estimate number of 'signal' factors as eigenvalues above MP bulk edge.
    """
    lam = eigenvalues.values
    if sigma2 is None:
        sigma2 = np.median(lam)
    threshold = mp_bulk_edge(q=q, sigma2=sigma2)
    n_factors = int((lam > threshold).sum())
    return max(n_factors, 0)

def denoise_constant_residual_eigenvalue(eigenvalues: pd.Series, eigenvectors: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    """
    Replace all residual eigenvalues (beyond top n_factors) by their average, rebuild correlation, renormalize.
    """
    lam = eigenvalues.values.copy()
    if n_factors < len(lam):
        lam[n_factors:] = lam[n_factors:].mean()
    Lambda = np.diag(lam)
    corr = eigenvectors.values @ Lambda @ eigenvectors.values.T
    diag = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    inv_diag = np.diag(1.0 / diag)
    corr = inv_diag @ corr @ inv_diag
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=eigenvectors.index, columns=eigenvectors.index)

def denoise_target_shrinkage(eigenvalues: pd.Series, eigenvectors: pd.DataFrame, n_factors: int, alpha: float = 0.0) -> pd.DataFrame:
    """
    Shrink the residual subspace toward its diagonal:
        C_residual_shrunk = alpha * C_residual + (1-alpha) * diag(C_residual)
    Keep signal subspace intact. Return renormalized correlation.
    """
    lam = eigenvalues.values
    V = eigenvectors.values
    V_sig, lam_sig = V[:, :n_factors], lam[:n_factors]
    V_res, lam_res = V[:, n_factors:], lam[n_factors:]

    signal_cov = V_sig @ np.diag(lam_sig) @ V_sig.T
    residual_cov = V_res @ np.diag(lam_res) @ V_res.T
    residual_shrunk = alpha * residual_cov + (1.0 - alpha) * np.diag(np.diag(residual_cov))
    corr = signal_cov + residual_shrunk

    diag = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    inv_diag = np.diag(1.0 / diag)
    corr = inv_diag @ corr @ inv_diag
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=eigenvectors.index, columns=eigenvectors.index)

def remove_market_mode(correlation_matrix: pd.DataFrame, eigenvalues: pd.Series, eigenvectors: pd.DataFrame, n_market_components: int = 1) -> pd.DataFrame:
    """
    Remove the first n_market_components principal components ('detoning'), then renormalize to correlation.
    """
    lam = eigenvalues.values
    V = eigenvectors.values
    Vm = V[:, :n_market_components]
    lam_m = lam[:n_market_components]
    market_cov = Vm @ np.diag(lam_m) @ Vm.T

    corr = correlation_matrix.values.copy()
    corr = corr - market_cov
    diag = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    inv_diag = np.diag(1.0 / diag)
    corr = inv_diag @ corr @ inv_diag
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=correlation_matrix.index, columns=correlation_matrix.columns)

def denoise_covariance_with_mp(
    covariance_matrix: pd.DataFrame,
    q: float,
    method: str = "constant_residual_eigenvalue",
    alpha: float = 0.0,
    remove_market: bool = False,
    n_market_components: int = 1,
) -> pd.DataFrame:
    """
    End-to-end RMT(MP) denoising on a covariance matrix.
    """
    std = pd.Series(np.sqrt(np.diag(covariance_matrix.values)), index=covariance_matrix.index)
    inv_std = np.diag(1.0 / np.clip(std.values, 1e-12, None))
    correlation = pd.DataFrame(inv_std @ covariance_matrix.values @ inv_std,
                               index=covariance_matrix.index, columns=covariance_matrix.columns)
    np.fill_diagonal(correlation.values, 1.0)

    eigenvalues, eigenvectors = eigen_decomposition(correlation)
    n_factors = count_signal_factors(eigenvalues, q=q, sigma2=None)

    if method == "constant_residual_eigenvalue":
        corr_denoised = denoise_constant_residual_eigenvalue(eigenvalues, eigenvectors, n_factors)
    elif method == "target_shrinkage":
        corr_denoised = denoise_target_shrinkage(eigenvalues, eigenvectors, n_factors, alpha=alpha)
    else:
        raise ValueError("method must be one of {'constant_residual_eigenvalue','target_shrinkage'}")

    if remove_market and n_market_components > 0:
        corr_denoised = remove_market_mode(corr_denoised, eigenvalues, eigenvectors, n_market_components=n_market_components)

    std_mat = np.diag(std.values)
    cov_denoised = pd.DataFrame(std_mat @ corr_denoised.values @ std_mat, index=std.index, columns=std.index)
    return cov_denoised

def denoise_covariance(
    covariance_matrix: pd.DataFrame,
    q: float,
    method: str = "constant_residual_eigenvalue",
    alpha: float = 0.0,
    remove_market: bool = False,
    n_market_components: int = 1,
) -> pd.DataFrame:
    """
    Convenience wrapper for denoise_covariance_with_mp (same arguments).
    """
    return denoise_covariance_with_mp(
        covariance_matrix=covariance_matrix,
        q=q,
        method=method,
        alpha=alpha,
        remove_market=remove_market,
        n_market_components=n_market_components,
    )
