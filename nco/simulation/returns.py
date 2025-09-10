from __future__ import annotations
import numpy as np
import pandas as pd

def _student_t_shocks(n_obs: int, dof: float | None, rng: np.random.Generator) -> np.ndarray:
    if dof is None or dof <= 0:
        return rng.standard_normal(n_obs)
    z = rng.standard_normal(n_obs)
    w = rng.chisquare(dof, size=n_obs) / dof
    return z / np.sqrt(w)

def arma_filter(
    innovations: np.ndarray,
    ar_coefficients: np.ndarray | list[float] | tuple[float, ...] = (),
    ma_coefficients: np.ndarray | list[float] | tuple[float, ...] = (),
    constant: float = 0.0,
) -> np.ndarray:
    """
    ARMA(p,q) mean filter.

    Model:
        x_t = constant
              + sum_{i=1..p} phi_i * x_{t-i}
              + innovations_t
              + sum_{j=1..q} theta_j * innovations_{t-j}

    Args:
        innovations: 1D array of white-noise shocks (len T). This is ε_t.
        ar_coefficients: sequence of AR coefficients [phi_1, ..., phi_p].
        ma_coefficients: sequence of MA coefficients [theta_1, ..., theta_q].
        constant: intercept term (c).

    Returns:
        mean_component: 1D array (len T) of x_t satisfying the ARMA recursion.
                        (You typically add this to a volatility-driven component.)
    """
    eps = np.asarray(innovations, dtype=float).ravel()
    phi = np.asarray(ar_coefficients, dtype=float).ravel()
    theta = np.asarray(ma_coefficients, dtype=float).ravel()
    p, q = len(phi), len(theta)
    T = eps.shape[0]

    mean_component = np.zeros(T, dtype=float)
    # We keep a small buffer for past eps to avoid if-branches in the loop
    eps_padded = np.concatenate([np.zeros(q, dtype=float), eps])

    for t in range(T):
        # AR part: depend on past x
        ar_part = 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                ar_part += phi[i - 1] * mean_component[t - i]
        # MA part: depend on past eps
        ma_part = 0.0
        if q > 0:
            # eps index in padded is (t + q); past j steps is (t + q - j)
            for j in range(1, q + 1):
                ma_part += theta[j - 1] * eps_padded[t + q - j]

        mean_component[t] = constant + ar_part + eps[t] + ma_part

    return mean_component

def garch_filter(
    standardized_shocks: np.ndarray,
    omega: float,
    alpha_coefficients: np.ndarray | list[float] | tuple[float, ...] = (0.05,),
    beta_coefficients:  np.ndarray | list[float] | tuple[float, ...] = (0.94,),
    initial_variance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GARCH(p,q) conditional variance filter with arbitrary orders.

    Model:
        h_t = omega
              + sum_{i=1..p} alpha_i * r_{t-i}^2
              + sum_{j=1..q} beta_j  * h_{t-j}
        r_t = sqrt(h_t) * z_t    (z_t are standardized shocks, mean 0 var 1)

    Args:
        standardized_shocks: 1D array (len T) of z_t (approximately N(0,1) or t-standardized).
        omega: long-run variance intercept (>= 0).
        alpha_coefficients: sequence [alpha_1, ..., alpha_p] for lagged squared returns.
        beta_coefficients:  sequence [beta_1,  ..., beta_q] for lagged variance.
        initial_variance: optional h_0. If None, uses omega / (1 - sum(alpha) - sum(beta))
                          when the denominator is positive; otherwise falls back to variance of shocks.

    Returns:
        garch_returns: 1D array r_t = sqrt(h_t) * z_t  (the volatility-driven component)
        conditional_variance: 1D array h_t
    """
    z = np.asarray(standardized_shocks, dtype=float).ravel()
    alpha = np.asarray(alpha_coefficients, dtype=float).ravel()
    beta  = np.asarray(beta_coefficients,  dtype=float).ravel()
    p, q = len(alpha), len(beta)
    T = z.shape[0]

    alpha_sum = float(alpha.sum()) if p > 0 else 0.0
    beta_sum  = float(beta.sum())  if q > 0 else 0.0
    denom = max(1e-12, 1.0 - alpha_sum - beta_sum)

    if initial_variance is None:
        if denom > 1e-6:
            h_prev = omega / denom
        else:
            # fallback: empirical variance of z scaled by omega as a guard
            h_prev = max(1e-6, float(z.var(ddof=1)))
    else:
        h_prev = float(initial_variance)

    conditional_variance = np.zeros(T, dtype=float)
    garch_returns = np.zeros(T, dtype=float)
    r_squared_history = np.zeros(max(1, p), dtype=float)  # store last p values of r_{t}^2
    h_history         = np.full(max(1, q), h_prev, dtype=float)  # store last q values of h_{t}

    for t in range(T):
        arq_part = 0.0
        if p > 0:
            # match alpha_i with r_{t-i}^2; r_squared_history[0] = r_{t-1}^2, etc.
            for i in range(p):
                if i < t:  # there exists r_{t-1-i}
                    arq_part += alpha[i] * r_squared_history[i]
        garch_part = 0.0
        if q > 0:
            for j in range(q):
                garch_part += beta[j] * h_history[j]

        h_t = omega + arq_part + garch_part
        h_t = float(max(h_t, 1e-12))  # numerical guard

        r_t = np.sqrt(h_t) * z[t]

        # shift histories (latest first)
        if p > 0:
            r_squared_history = np.roll(r_squared_history, 1)
            r_squared_history[0] = r_t ** 2
        if q > 0:
            h_history = np.roll(h_history, 1)
            h_history[0] = h_t

        conditional_variance[t] = h_t
        garch_returns[t] = r_t

    return garch_returns, conditional_variance

def simulate_arma_garch_block_returns(
    asset_names: list[str],
    n_observations: int,
    block_correlation: pd.DataFrame,
    # ARMA mean
    ar_coefficients: list[float] | tuple[float, ...] | dict[str, list[float] | tuple[float, ...]] = (),
    ma_coefficients: list[float] | tuple[float, ...] | dict[str, list[float] | tuple[float, ...]] = (),
    arma_constant: float | dict[str, float] = 0.0,
    # GARCH variance
    omega: float | dict[str, float] = 1e-6,
    alpha_coefficients: list[float] | tuple[float, ...] | dict[str, list[float] | tuple[float, ...]] = (0.05,),
    beta_coefficients:  list[float] | tuple[float, ...] | dict[str, list[float] | tuple[float, ...]] = (0.94,),
    initial_variance: float | dict[str, float] | None = None,
    # shocks
    dof_common_shock: float | None = 7.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate returns with cross-sectional dependence from a TARGET correlation (e.g., block),
    ARMA(p,q) mean per asset, and GARCH(p,q) conditional variance per asset.

    Steps per t:
      1) draw standardized common shocks z_t ~ (t or Gaussian) with corr = R_block
      2) per-asset GARCH(p,q): r_vol_{i,t} = sqrt(h_{i,t}) * z_{i,t}
      3) per-asset ARMA(p,q) mean: m_{i,t} from own z_{i,t}
      4) return r_{i,t} = m_{i,t} + r_vol_{i,t}
    """
    rng = np.random.default_rng(seed)
    assets = list(asset_names)
    N = len(assets)
    T = int(n_observations)

    # ----- params per asset -----
    ar_dict   = broadcast_param(assets, ar_coefficients,  (), is_coeff_sequence=True)
    ma_dict   = broadcast_param(assets, ma_coefficients,  (), is_coeff_sequence=True)
    c_dict    = broadcast_param(assets, arma_constant,    0.0)
    omega_dict= broadcast_param(assets, omega,            1e-6)
    alpha_dict= broadcast_param(assets, alpha_coefficients, (0.05,), is_coeff_sequence=True)
    beta_dict = broadcast_param(assets, beta_coefficients,  (0.94,), is_coeff_sequence=True)
    if isinstance(initial_variance, dict) or initial_variance is None:
        h0_dict = broadcast_param(assets, initial_variance, None)
    else:
        h0_dict = broadcast_param(assets, initial_variance, float(initial_variance))

    # ----- correlated common shocks (T x N) -----
    R = block_correlation.loc[assets, assets].values
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    # Cholesky with jitter
    jitter = 1e-12
    for _ in range(6):
        try:
            L = np.linalg.cholesky(R + jitter * np.eye(N))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        # eigen clip fallback
        eigval, eigvec = np.linalg.eigh(R)
        eigval = np.clip(eigval, 1e-10, None)
        R = eigvec @ np.diag(eigval) @ eigvec.T
        L = np.linalg.cholesky(R)

    iid = np.vstack([_student_t_shocks(T, dof_common_shock, rng) for _ in range(N)]).T  # T x N
    shocks = iid @ L.T
    # standardize each column to mean0, var1 (robustness)
    shocks = (shocks - shocks.mean(axis=0)) / (shocks.std(axis=0, ddof=1) + 1e-12)

    # ----- per-asset filters -----
    arma_mean = np.zeros((T, N))
    garch_ret = np.zeros((T, N))
    cond_var  = np.zeros((T, N))

    for j, a in enumerate(assets):
        z_j = shocks[:, j]
        # variance
        r_vol, h_path = garch_filter(
            standardized_shocks=z_j,
            omega=float(omega_dict[a]),
            alpha_coefficients=alpha_dict[a],
            beta_coefficients=beta_dict[a],
            initial_variance=h0_dict[a],
        )
        # mean
        m_j = arma_filter(
            innovations=z_j,
            ar_coefficients=ar_dict[a],
            ma_coefficients=ma_dict[a],
            constant=float(c_dict[a]),
        )

        arma_mean[:, j] = m_j
        garch_ret[:, j] = r_vol
        cond_var[:, j]  = h_path

    returns = arma_mean + garch_ret
    returns_df = pd.DataFrame(returns, columns=assets)

    info = {
        "arma_mean": pd.DataFrame(arma_mean, columns=assets),
        "garch_innov": pd.DataFrame(garch_ret, columns=assets),
        "conditional_std": pd.DataFrame(np.sqrt(cond_var), columns=assets),
        "common_shocks": pd.DataFrame(shocks, columns=assets),
        "params": {
            "ar": ar_dict, "ma": ma_dict, "constant": c_dict,
            "omega": omega_dict, "alpha": alpha_dict, "beta": beta_dict,
            "dof_common_shock": dof_common_shock,
        }
    }
    return returns_df, info