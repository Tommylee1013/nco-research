from __future__ import annotations
import numpy as np
import pandas as pd

from numpy.linalg import eigh
from typing import Tuple
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

def get_pca(matrix : pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    get principal component analysis matrix
    :param matrix: covariance matrix
    :return:
    """
    eVal, eVec = np.linalg.eig(matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec

def marchenko_pastur_prob_distribution(var, q, pts) :
    eMin, eMax = var * (1 - (1.0 / q) ** 0.5) ** 2, var * (1 + (1.0 / q) ** 0.5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    pdf = pd.Series(pdf, index = eVal)
    return pdf

def fit_kde(obs, bWidth = 0.25, kernel = 'gaussian', x = None) :
    if len(obs.shape) == 1: obs = obs.reshape(-1, 1)
    kde = KernelDensity(
        kernel = kernel,
        bandwidth = bWidth
    ).fit(obs)

    if x is None: x = np.unique(obs).reshape(-1, 1)

    if len(x.shape) == 1: x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(
        np.exp(logProb),
        index = x.flatten()
    )
    return pdf

def pdf_error(var, eVal, q, bWidth, pts = 1000, verbose = False) :
    var = var[0]
    pdf0 = marchenko_pastur_prob_distribution(var, q, pts)  # theoretical pdf
    pdf1 = fit_kde(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    if verbose : print("sse:" + str(sse))
    return sse

def find_max_eval(eVal, q, bWidth, verbose=False):
    out = minimize(
        lambda *x: pdf_error(*x),
        x0=np.array(0.5),
        args=(eVal, q, bWidth),
        bounds=((1E-5, 1 - 1E-5),)
    )

    if verbose: print("found errPDFs" + str(out['x'][0]))

    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / q) ** .5) ** 2

    return eMax, var

def symmetrize(matrix: pd.DataFrame) -> pd.DataFrame :
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
    matrix = symmetrize(matrix.copy())
    idx = matrix.index
    for _ in range(max_iter):
        eigenvalues, eigenvectors = eigh(matrix.values)
        if np.all(eigenvalues >= eps):
            break
        eigenvalues = np.clip(eigenvalues, eps, None)
        matrix = pd.DataFrame((eigenvectors * eigenvalues) @ eigenvectors.T, index=idx, columns=idx)
        matrix = symmetrize(matrix)
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
    correlation_matrix = symmetrize(correlation_matrix.copy())
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
    return symmetrize(result)

def covariance_to_correlation(
        cov_matrix : pd.DataFrame,
        eps : float = 1e-12
    ) -> pd.DataFrame :
    """
    Convert covariance matrix to correlation matrix
    :param cov_matrix: covariance matrix from asset returns
    :param eps: epsilon, default 1e-12
    :return: correlation matrix
    """
    s = np.sqrt(
        np.clip(
            np.diag(cov_matrix.values), eps, None
        )
    )
    d_inv = np.diag(1 / s)
    result = pd.DataFrame(
        d_inv @ cov_matrix.values @ d_inv,
        index = cov_matrix.index,
        columns = cov_matrix.columns
    )
    np.fill_diagonal(result.values, 1)
    return symmetrize(result)

def correlation_to_covariance(
        corr_matrix : pd.DataFrame,
        volatility : pd.Series
    ) -> pd.DataFrame :
    """
    Convert correlation matrix to covariance matrix given volatility vector
    :param corr_matrix: correlation matrix from asset returns'
    :param volatility: volatility series
    :return: covariance matrix
    """
    d = np.diag(volatility.values)
    res = pd.DataFrame(
        d @ corr_matrix.values @ d,
        index = corr_matrix.index,
        columns = corr_matrix.columns
    )

    return res

def correlation_distance(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation distance used in HRP/NCO:
        d_ij = sqrt((1 - rho_ij) / 2)
    """
    distance = np.sqrt((1.0 - correlation_matrix) * 0.5)
    np.fill_diagonal(distance.values, 0.0)
    return distance