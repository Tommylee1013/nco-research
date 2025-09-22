from __future__ import annotations
import argparse
import os
from datetime import datetime
import pandas as pd

from nco.experiment.montecarlo import monte_carlo_experiment

def _parse_floats(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip() != ""]

def parse_args():
    p = argparse.ArgumentParser(
        description="Run Monte Carlo comparing Markowitz / NCO / Posterior-NCO / IVP "
                    "with Ledoit–Wolf shrinkage, MP denoising, and optional managers' views."
    )

    # -------- Core problem size / data-gen --------
    p.add_argument("--n_assets", type=int, default=60, help="Number of assets.")
    p.add_argument("--n_clusters", type=int, default=5, help="Number of clusters in the true block structure.")
    p.add_argument("--n_in_sample", type=int, default=252, help="In-sample length (T_in).")
    p.add_argument("--n_out_of_sample", type=int, default=252, help="Out-of-sample length (T_out).")
    p.add_argument("--n_trials", type=int, default=200, help="Monte Carlo trials.")
    p.add_argument("--rho_in", type=float, default=0.6, help="True intra-cluster correlation.")
    p.add_argument("--rho_out", type=float, default=0.1, help="True inter-cluster correlation.")
    p.add_argument("--df", type=float, default=5.0,
                   help="Student-t degrees of freedom (<=0 → Gaussian).")
    p.add_argument("--rf", type=float, default=0.0, help="Risk-free rate.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # -------- Estimation: shrinkage / denoising --------
    p.add_argument("--shrinkage", type=str, default="lw_constant_corr",
                   choices=["lw_identity", "lw_constant_corr", "none"],
                   help="Ledoit–Wolf shrinkage type.")
    p.add_argument("--denoising", type=str, default="none",
                   choices=["none", "mp_constant", "mp_target"],
                   help="RMT denoising method.")
    p.add_argument("--denoising_alpha", type=float, default=0.0,
                   help="Residual blend alpha for mp_target.")
    p.add_argument("--detone", action="store_true",
                   help="Remove dominant market component(s) during denoising.")
    p.add_argument("--n_market_components", type=int, default=1,
                   help="Number of market components to remove if --detone is set.")

    # -------- Managers' Views (Posterior-NCO) --------
    p.add_argument("--use_views", action="store_true",
                   help="Enable managers' views to produce Posterior-NCO.")
    p.add_argument("--view_branch", type=str, default="black_litterman",
                   choices=["black_litterman", "corr_blend"],
                   help="Use Black–Litterman posterior or correlation-view blending.")
    # BL view config
    p.add_argument("--bl_view_type", type=str, default="pairwise",
                   choices=["pairwise", "absolute", "cluster"],
                   help="Type of BL views to generate.")
    p.add_argument("--n_views", type=int, default=10, help="Number of views.")
    p.add_argument("--view_noise_std", type=float, default=1e-4, help="Std of noise added to q.")
    p.add_argument("--view_misspec_probability", type=float, default=0.0,
                   help="Probability to flip a view sign (mis-specification).")
    p.add_argument("--view_confidence_scale", type=float, default=0.1,
                   help="Scale for Omega = scale * diag(P Σ P^T). Lower = stronger views.")
    # Correlation-blend config
    p.add_argument("--beta_view", type=float, default=0.3,
                   help="Mix weight for correlation blending (posterior = (1-β)R_hat + βR_view).")
    p.add_argument("--intra_scale_view", type=float, default=1.0,
                   help="Scale factor for intra-cluster correlations in R_view.")
    p.add_argument("--inter_scale_view", type=float, default=1.0,
                   help="Scale factor for inter-cluster correlations in R_view.")
    p.add_argument("--corr_view_noise_std", type=float, default=0.0,
                   help="Noise std added to correlation view before projection.")

    # -------- Simulator (optional) --------
    p.add_argument("--simulator", type=str, default="gaussian",
                   choices=["gaussian", "arma_garch_block"],
                   help="Data generator for in/out samples.")
    p.add_argument("--arma_ar", type=str, default="",
                   help="Comma-separated AR coefficients for ARMA(p,q), e.g. '0.3,-0.1'.")
    p.add_argument("--arma_ma", type=str, default="",
                   help="Comma-separated MA coefficients for ARMA(p,q), e.g. '0.2'.")
    p.add_argument("--arma_constant", type=float, default=0.0,
                   help="ARMA intercept term.")
    p.add_argument("--garch_omega", type=float, default=1e-6,
                   help="GARCH intercept omega.")
    p.add_argument("--garch_alpha", type=str, default="0.05",
                   help="Comma-separated alpha coefficients for GARCH(p,q), e.g. '0.06,0.03'.")
    p.add_argument("--garch_beta", type=str, default="0.94",
                   help="Comma-separated beta coefficients for GARCH(p,q), e.g. '0.90'.")
    p.add_argument("--garch_h0", type=float, default=None,
                   help="Initial variance h0 (optional).")
    p.add_argument("--common_shock_df", type=float, default=7.0,
                   help="t dof for common shocks in arma_garch_block (<=0 for Gaussian).")

    # -------- IO --------
    p.add_argument("--outdir", type=str, default="data/synthetic", help="Output directory for CSV.")
    p.add_argument("--filename_prefix", type=str, default="mc",
                   help="Prefix for output filename (without extension).")

    return p.parse_args()


def main():
    args = parse_args()

    # Interpret df <= 0 as Gaussian
    df = None if args.df is None or args.df <= 0 else float(args.df)
    common_df = None if args.common_shock_df is None or args.common_shock_df <= 0 else float(args.common_shock_df)

    df_results = monte_carlo_experiment(
        # core
        n_assets=args.n_assets,
        n_clusters=args.n_clusters,
        n_in_sample=args.n_in_sample,
        n_out_of_sample=args.n_out_of_sample,
        n_trials=args.n_trials,
        intra_cluster_correlation=args.rho_in,
        inter_cluster_correlation=args.rho_out,
        degrees_of_freedom=df,
        risk_free_rate=args.rf,
        seed=args.seed,
        # estimation
        shrinkage=args.shrinkage,
        denoising_method=args.denoising,
        denoising_alpha=args.denoising_alpha,
        detone=args.detone,
        n_market_components=args.n_market_components,
        # views
        use_views=args.use_views,
        view_branch=args.view_branch,
        bl_view_type=args.bl_view_type,
        n_views=args.n_views,
        view_noise_std=args.view_noise_std,
        view_misspec_probability=args.view_misspec_probability,
        view_confidence_scale=args.view_confidence_scale,
        beta_view=args.beta_view,
        intra_scale_view=args.intra_scale_view,
        inter_scale_view=args.inter_scale_view,
        corr_view_noise_std=args.corr_view_noise_std,
        # simulator
        simulator=args.simulator,
        arma_ar=_parse_floats(args.arma_ar),
        arma_ma=_parse_floats(args.arma_ma),
        arma_constant=args.arma_constant,
        garch_omega=args.garch_omega,
        garch_alpha=_parse_floats(args.garch_alpha),
        garch_beta=_parse_floats(args.garch_beta),
        garch_h0=args.garch_h0,
        common_shock_df=common_df,
    )

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [
        args.filename_prefix,
        f"SIM{args.simulator}",
        f"N{args.n_assets}",
        f"T{args.n_in_sample}-{args.n_out_of_sample}",
        f"K{args.n_clusters}",
        args.shrinkage,
        args.denoising + ("_detone" if args.detone else ""),
    ]
    if args.use_views:
        if args.view_branch == "black_litterman":
            parts += [f"BL-{args.bl_view_type}", f"V{args.n_views}", f"Omega{args.view_confidence_scale:g}"]
        else:
            parts += [f"CBeta{args.beta_view:g}", f"intra{args.intra_scale_view:g}", f"inter{args.inter_scale_view:g}"]
    fname = "_".join(parts) + f"_{ts}.csv"
    out_path = os.path.join(args.outdir, fname)

    df_results.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Optional quick summary to stdout
    try:
        summary = df_results.groupby("method")[["sharpe", "vol", "mdd", "hhi"]].agg(["mean", "median"])
        pd.set_option("display.width", 120)
        print("\n==== Summary (mean/median) ====\n", summary)
    except Exception:
        pass

if __name__ == "__main__":
    main()