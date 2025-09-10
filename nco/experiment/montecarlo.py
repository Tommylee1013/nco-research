# nco/experiment/montecarlo.py

"""
Monte Carlo experiment comparing Markowitz, NCO, Posterior-NCO, and IVP.

Pipeline (per trial):
  1) Generate block-structured "true" covariance/correlation and ground-truth expected returns
  2) Simulate in-sample / out-of-sample returns (Gaussian or Student-t)
  3) Estimate in-sample expected returns and covariance
  4) Apply Ledoit–Wolf shrinkage (identity or constant-correlation)
  5) (Optional) Apply Marčenko–Pastur (RMT) denoising and/or detoning
  6) Compute two branches:
       - Baseline branch (no views): Markowitz, NCO, IVP
       - Posterior branch (with managers' views):
            (A) Black–Litterman posterior on mean/covariance  OR
            (B) Correlation-view blending, then rebuild covariance
         -> Posterior correlation -> distance -> NCO  ==> "Posterior-NCO"
  7) Evaluate all portfolios on out-of-sample returns

All inputs/outputs use pandas Series/DataFrame and preserve index/columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..optimization.utils import (
    covariance_to_correlation,
    correlation_to_covariance,
    nearest_correlation,
)
from ..optimization.shrinkage import (
    ledoit_wolf_shrinkage_identity,
    ledoit_wolf_shrinkage_constant_correlation,
)
from ..optimization.denoising import denoise_covariance_with_mp
from ..optimization.markowitz import (
    maximum_sharpe_portfolio,
    inverse_variance_portfolio,
)
from ..optimization.nco import nco_portfolio

from ..simulation.generator import generate_block_covariance, simulate_returns
from ..simulation.metrics import evaluate_portfolio
from ..simulation.returns import simulate_arma_garch_block_returns


from ..bayesian.black_litterman import black_litterman_posterior
from ..bayesian.views import (
    build_pairwise_relative_views,
    build_absolute_views,
    build_cluster_relative_views,
    calibrate_view_uncertainty,
)
from ..optimization.corr_views import make_correlation_view_from_true


def _apply_shrinkage(
    in_sample_returns: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Apply Ledoit–Wolf shrinkage to the sample covariance.

    Args:
        in_sample_returns: DataFrame (T_in x N)
        method: one of {'lw_identity', 'lw_constant_corr', 'none'}

    Returns:
        shrunk_covariance: DataFrame (N x N)
    """
    if method == "lw_identity":
        return ledoit_wolf_shrinkage_identity(in_sample_returns)
    if method == "lw_constant_corr":
        return ledoit_wolf_shrinkage_constant_correlation(in_sample_returns)
    if method == "none":
        n_assets = in_sample_returns.shape[1]
        cov = np.cov(in_sample_returns.values, rowvar=False)
        return pd.DataFrame(cov, index=in_sample_returns.columns, columns=in_sample_returns.columns)
    raise ValueError("shrinkage method must be one of {'lw_identity','lw_constant_corr','none'}")


def _apply_denoising(
    covariance_matrix: pd.DataFrame,
    q_ratio: float,
    method: str,
    alpha: float,
    detone: bool,
    n_market_components: int,
) -> pd.DataFrame:
    """
    Optionally denoise the covariance with Marčenko–Pastur and detoning.

    Args:
        covariance_matrix: prior covariance (after shrinkage)
        q_ratio: T_in / N
        method: {'none','mp_constant','mp_target'}
        alpha: residual blend parameter for 'mp_target'
        detone: whether to remove market components
        n_market_components: number of market PCs to remove

    Returns:
        denoised_covariance
    """
    if method == "none":
        return covariance_matrix
    mp_method = "constant_residual_eigenvalue" if method == "mp_constant" else "target_shrinkage"
    return denoise_covariance_with_mp(
        covariance_matrix=covariance_matrix,
        q=q_ratio,
        method=mp_method,
        alpha=alpha,
        remove_market=detone,
        n_market_components=n_market_components,
    )


def _apply_black_litterman_views(
    prior_expected_returns: pd.Series,
    prior_covariance: pd.DataFrame,
    asset_names: list[str],
    rng: np.random.Generator,
    view_type: str = "pairwise",     # {'pairwise','absolute','cluster'}
    n_clusters: int = 5,
    n_views: int = 10,
    noise_std: float = 1e-4,
    mis_spec_probability: float = 0.0,
    confidence_scale: float = 1.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Construct BL views and compute the posterior (μ*, Σ*).
    The base input should already be stabilized (shrinkage/denoising).

    Returns:
        posterior_expected_returns, posterior_covariance
    """
    if view_type == "pairwise":
        P, q, Omega = build_pairwise_relative_views(
            asset_names=asset_names,
            base_expected_returns=prior_expected_returns,
            n_views=n_views,
            noise_std=noise_std,
            mis_spec_probability=mis_spec_probability,
            rng=rng,
        )
    elif view_type == "absolute":
        P, q, Omega = build_absolute_views(
            asset_names=asset_names,
            base_expected_returns=prior_expected_returns,
            n_views=n_views,
            noise_std=noise_std,
            mis_spec_probability=mis_spec_probability,
            rng=rng,
        )
    elif view_type == "cluster":
        P, q, Omega, _ = build_cluster_relative_views(
            asset_names=asset_names,
            n_clusters=n_clusters,
            base_expected_returns=prior_expected_returns,
            n_views=min(n_views, max(1, n_clusters - 1)),
            noise_std=noise_std,
            mis_spec_probability=mis_spec_probability,
            rng=rng,
        )
    else:
        raise ValueError("view_type must be one of {'pairwise','absolute','cluster'}")

    # Calibrate Ω to prior covariance scale
    Omega = calibrate_view_uncertainty(P, prior_covariance, scale=confidence_scale)

    posterior_mu, posterior_cov = black_litterman_posterior(
        prior_expected_returns=prior_expected_returns,
        prior_covariance=prior_covariance,
        view_matrix=P,
        view_values=q,
        view_uncertainty=Omega,
    )
    return posterior_mu, posterior_cov


def _apply_correlation_view_blending(
    prior_covariance: pd.DataFrame,
    true_correlation_for_view: pd.DataFrame,
    rng: np.random.Generator,
    beta_view: float = 0.3,
    intra_scale_view: float = 1.0,
    inter_scale_view: float = 1.0,
    corr_view_noise_std: float = 0.0,
) -> pd.DataFrame:
    """
    Build a correlation 'view' and blend with the empirical correlation, then
    map back to covariance using the prior volatilities.

    Returns:
        posterior_covariance
    """
    # Prior correlation from prior covariance (already shrunk/denoised)
    prior_correlation = covariance_to_correlation(prior_covariance)
    # A plausible correlation 'view' around the truth (sim setup)
    correlation_view = make_correlation_view_from_true(
        true_correlation=true_correlation_for_view,
        intra_scale=intra_scale_view,
        inter_scale=inter_scale_view,
        noise_std=corr_view_noise_std,
        rng=rng,
    )
    posterior_correlation = nearest_correlation(
        (1.0 - beta_view) * prior_correlation + beta_view * correlation_view
    )
    # Keep prior volatilities
    vol_prior = pd.Series(np.sqrt(np.diag(prior_covariance.values)), index=prior_covariance.index)
    posterior_covariance = correlation_to_covariance(posterior_correlation, vol_prior)
    return posterior_covariance


def monte_carlo_experiment(
    asset_names: list[str] | None = None,
    n_assets: int = 60,
    n_clusters: int = 5,
    n_in_sample: int = 252,
    n_out_of_sample: int = 252,
    n_trials: int = 200,
    intra_cluster_correlation: float = 0.6,
    inter_cluster_correlation: float = 0.1,
    degrees_of_freedom: float | None = 5.0,
    shrinkage: str = "lw_constant_corr",      # {'lw_identity','lw_constant_corr','none'}
    denoising_method: str = "none",           # {'none','mp_constant','mp_target'}
    denoising_alpha: float = 0.0,
    detone: bool = False,
    n_market_components: int = 1,
    # Managers' Views (to build Posterior-NCO)
    use_views: bool = True,
    view_branch: str = "black_litterman",     # {'black_litterman','corr_blend'}
    # BL view config
    bl_view_type: str = "pairwise",           # {'pairwise','absolute','cluster'}
    n_views: int = 10,
    view_noise_std: float = 1e-4,
    view_misspec_probability: float = 0.0,
    view_confidence_scale: float = 1.0,
    # Corr-blend view config
    beta_view: float = 0.3,
    intra_scale_view: float = 1.0,
    inter_scale_view: float = 1.0,
    corr_view_noise_std: float = 0.0,
    # Evaluation
    risk_free_rate: float = 0.0,
    seed: int | None = 123,
    # Simulator choice
    simulator: str = "gaussian",   # {'gaussian', 'arma_garch_block'}
    # ARMA–GARCH params (only used if simulator == 'arma_garch_block')
    arma_ar: list[float] | tuple[float, ...] = (),
    arma_ma: list[float] | tuple[float, ...] = (),
    arma_constant: float = 0.0,
    garch_omega: float = 1e-6,
    garch_alpha: list[float] | tuple[float, ...] = (0.05,),
    garch_beta:  list[float] | tuple[float, ...] = (0.94,),
    garch_h0: float | None = None,
    common_shock_df: float | None = 7.0,
) -> pd.DataFrame:
    """
    Run a Monte Carlo study comparing four methods:
      - 'Markowitz'         : long-only max Sharpe on stabilized covariance
      - 'NCO'               : NCO on stabilized correlation (no views)
      - 'Posterior-NCO'     : NCO on posterior correlation after managers' views
      - 'IVP'               : inverse-variance portfolio on stabilized covariance

    "Stabilized" = after Ledoit–Wolf shrinkage (+ optional MP denoising/detoning).
    Posterior-NCO = apply views (BL or correlation blending) AFTER stabilization.

    Returns:
        DataFrame with metrics for each method and trial.
    """
    # Names
    if asset_names is None:
        asset_names = [f"Asset{i:03d}" for i in range(n_assets)]
    else:
        n_assets = len(asset_names)

    rng = np.random.default_rng(seed)
    results: list[dict] = []

    for trial in range(n_trials):
        # 1) Truth generation (block structure) and ground-truth mean
        true_correlation, true_covariance = generate_block_covariance(
            asset_names=asset_names,
            n_clusters=n_clusters,
            intra_cluster_correlation=intra_cluster_correlation,
            inter_cluster_correlation=inter_cluster_correlation,
            seed=rng.integers(1e9),
        )
        true_expected_returns = pd.Series(
            rng.normal(loc=0.05 / 252, scale=0.10 / np.sqrt(252), size=n_assets),
            index=asset_names,
            name="expected_return",
        )

        # 2) Simulate in-sample and out-of-sample returns
        # in_sample_returns = simulate_returns(
        #     expected_returns=true_expected_returns,
        #     covariance_matrix=true_covariance,
        #     n_observations=n_in_sample,
        #     degrees_of_freedom=degrees_of_freedom,
        #     seed=rng.integers(1e9),
        # )
        # out_of_sample_returns = simulate_returns(
        #     expected_returns=true_expected_returns,
        #     covariance_matrix=true_covariance,
        #     n_observations=n_out_of_sample,
        #     degrees_of_freedom=degrees_of_freedom,
        #     seed=rng.integers(1e9),
        # )
        if simulator == "gaussian":
            in_sample_returns = simulate_returns(
                expected_returns=true_expected_returns,
                covariance_matrix=true_covariance,
                n_observations=n_in_sample,
                degrees_of_freedom=degrees_of_freedom,
                seed=rng.integers(1e9),
            )
            out_of_sample_returns = simulate_returns(
                expected_returns=true_expected_returns,
                covariance_matrix=true_covariance,
                n_observations=n_out_of_sample,
                degrees_of_freedom=degrees_of_freedom,
                seed=rng.integers(1e9),
            )
        elif simulator == "arma_garch_block":
            total = n_in_sample + n_out_of_sample
            all_returns, _ = simulate_arma_garch_block_returns(
                asset_names=asset_names,
                n_observations=total,
                block_correlation=true_correlation,  # ← 블록 구조 그대로 활용
                ar_coefficients=arma_ar,
                ma_coefficients=arma_ma,
                arma_constant=arma_constant,
                omega=garch_omega,
                alpha_coefficients=garch_alpha,
                beta_coefficients=garch_beta,
                initial_variance=garch_h0,
                dof_common_shock=common_shock_df,
                seed=rng.integers(1e9),
            )
            in_sample_returns = all_returns.iloc[:n_in_sample]
            out_of_sample_returns = all_returns.iloc[n_in_sample:n_in_sample + n_out_of_sample]
        else:
            raise ValueError("simulator must be one of {'gaussian', 'arma_garch_block'}")

        # 3) Estimators (means + shrinkage)
        estimated_expected_returns = in_sample_returns.mean(axis=0)
        covariance_estimate = _apply_shrinkage(in_sample_returns, method=shrinkage)

        # 4) (Optional) RMT denoising / detoning
        q_ratio = n_in_sample / n_assets
        covariance_estimate = _apply_denoising(
            covariance_matrix=covariance_estimate,
            q_ratio=q_ratio,
            method=denoising_method,
            alpha=denoising_alpha,
            detone=detone,
            n_market_components=n_market_components,
        )

        # ---------- BASELINE BRANCH (no views): Markowitz / NCO / IVP ----------
        correlation_estimate = covariance_to_correlation(covariance_estimate)
        weights_markowitz = maximum_sharpe_portfolio(
            expected_returns=estimated_expected_returns,
            covariance_matrix=covariance_estimate,
            risk_free_rate=risk_free_rate,
        )
        weights_nco = nco_portfolio(
            expected_returns=estimated_expected_returns,
            covariance_matrix=covariance_estimate,
            correlation_matrix=correlation_estimate,
        )
        weights_ivp = inverse_variance_portfolio(covariance_estimate)

        # Evaluate baselines
        for method_name, weights in [
            ("Markowitz", weights_markowitz),
            ("NCO", weights_nco),
            ("IVP", weights_ivp),
        ]:
            metrics = evaluate_portfolio(weights, out_of_sample_returns, risk_free_rate)
            metrics.update(
                {
                    "trial": trial,
                    "method": method_name,
                    "shrinkage": shrinkage,
                    "denoising": denoising_method,
                    "detone": detone,
                    "views": False,
                    "view_branch": "none",
                }
            )
            results.append(metrics)

        # ---------- POSTERIOR BRANCH (with managers' views) → Posterior-NCO ----------
        if use_views:
            if view_branch == "black_litterman":
                posterior_mu, posterior_cov = _apply_black_litterman_views(
                    prior_expected_returns=estimated_expected_returns,
                    prior_covariance=covariance_estimate,
                    asset_names=asset_names,
                    rng=rng,
                    view_type=bl_view_type,
                    n_clusters=n_clusters,
                    n_views=n_views,
                    noise_std=view_noise_std,
                    mis_spec_probability=view_misspec_probability,
                    confidence_scale=view_confidence_scale,
                )
                posterior_corr = covariance_to_correlation(posterior_cov)

            elif view_branch == "corr_blend":
                posterior_cov = _apply_correlation_view_blending(
                    prior_covariance=covariance_estimate,
                    true_correlation_for_view=true_correlation,  # sim-only convenience
                    rng=rng,
                    beta_view=beta_view,
                    intra_scale_view=intra_scale_view,
                    inter_scale_view=inter_scale_view,
                    corr_view_noise_std=corr_view_noise_std,
                )
                posterior_mu = estimated_expected_returns
                posterior_corr = covariance_to_correlation(posterior_cov)
            else:
                raise ValueError("view_branch must be one of {'black_litterman','corr_blend'}")

            weights_posterior_nco = nco_portfolio(
                expected_returns=posterior_mu,
                covariance_matrix=posterior_cov,
                correlation_matrix=posterior_corr,
            )
            metrics = evaluate_portfolio(weights_posterior_nco, out_of_sample_returns, risk_free_rate)
            metrics.update(
                {
                    "trial": trial,
                    "method": "Posterior-NCO",
                    "shrinkage": shrinkage,
                    "denoising": denoising_method,
                    "detone": detone,
                    "views": True,
                    "view_branch": view_branch,
                }
            )
            results.append(metrics)

    return pd.DataFrame(results)
