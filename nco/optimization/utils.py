from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import eigh

def _symmetrize(matrix: pd.DataFrame) -> pd.DataFrame :
    """make a matrix symmetric"""
    return 0.5 * (matrix + matrix.T)

def nearest_positive_semidefinite(
        matrix: pd.DataFrame,
        eps: float = 1e-10,
        max_iter: int = 100
    ) -> pd.DataFrame:
    """
    Project a matrix to the nearest positive semidefinite matrix using eigenvalue clipping.
    Keeps index/columns of the DataFrame.
    :param matrix: matrix to be projected
    :param eps: epsilon, default 1e-10
    :param max_iter: max iterations, default 100
    :return:
    """
    matrix = _symmetrize(matrix.copy())
    idx = matrix.index
    for _ in range(max_iter):
        eigenvalues, eigenvectors = eigh(matrix.values)
        if np.all(eigenvalues >= eps):
            break
        eigenvalues = np.clip(eigenvalues, eps, None)
        matrix = pd.DataFrame((eigenvectors * eigenvalues) @ eigenvectors.T, index=idx, columns=idx)
        matrix = _symmetrize(matrix)
    return matrix

def nearest_correlation(
        correlation_matrix: pd.DataFrame,
        eps: float = 1e-10
    ) -> pd.DataFrame:
    """
    Project a symmetric DataFrame to the nearest correlation matrix:
    - Positive semidefinite
    - Unit diagonal
    :param correlation_matrix: use correlation matrix
    :param eps: epsilon value, default 1e-10
    :return:
    """
    correlation_matrix = _symmetrize(correlation_matrix.copy())
    np.fill_diagonal(correlation_matrix.values, 1.0)
    correlation_matrix = nearest_positive_semidefinite(correlation_matrix, eps=eps)
    d = np.sqrt(np.clip(np.diag(correlation_matrix.values), eps, None))
    d_inv = np.diag(1.0 / d)
    result = pd.DataFrame(
        d_inv @ correlation_matrix.values @ d_inv,
        index=correlation_matrix.index,
        columns=correlation_matrix.columns
    )
    np.fill_diagonal(result.values, 1.0)
    return _symmetrize(result)