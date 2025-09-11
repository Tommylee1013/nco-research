from __future__ import annotations
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Analyze Monte Carlo results.")
    p.add_argument("--indir", type=str, default=None, help="Input directory (used with --pattern).")
    p.add_argument("--pattern", type=str, default=None, help="Glob pattern (used with --indir).")
    p.add_argument("--infiles", nargs="*", default=None, help="Explicit list of CSV files to analyze.")
    return p.parse_args()


def load_results(indir: str | None, pattern: str | None, infiles: list[str] | None) -> pd.DataFrame:
    if infiles:
        files = [os.path.abspath(f) for f in infiles]
    elif indir and pattern:
        files = sorted(glob.glob(os.path.join(indir, pattern)))
    else:
        raise ValueError("Must provide either --infiles or (--indir and --pattern).")

    if not files:
        raise FileNotFoundError("No CSV files found for analysis.")
    return pd.concat([pd.read_csv(f) for f in files], axis=0, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["shrinkage","denoising","detone","method"])[["sharpe","vol","mdd","hhi"]].agg(["mean","median"])


def head_to_head(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Compare Sharpe ratio between method pairs.
    Args:
        df: results DataFrame
        pairs: list of (method1, method2)
    Returns:
        DataFrame with win rates
    """
    results = []
    for m1, m2 in pairs:
        pivot = df.pivot_table(
            index=["trial","shrinkage","denoising","detone"],
            columns="method",
            values="sharpe",
            aggfunc="mean"
        )
        if m1 not in pivot.columns or m2 not in pivot.columns:
            continue
        pivot = pivot.dropna(subset=[m1, m2], how="any")
        win_rate = (pivot[m1] > pivot[m2]).groupby(level=[1,2,3]).mean().rename("win_rate")
        tmp = win_rate.reset_index()
        tmp["compare"] = f"{m1}_vs_{m2}"
        results.append(tmp)
    return pd.concat(results, axis=0, ignore_index=True)


def plot_sharpe_box(df: pd.DataFrame):
    ax = df.boxplot(column="sharpe", by="method")
    plt.title("Out-of-sample Sharpe by Method")
    plt.suptitle("")
    plt.grid(False)
    plt.axhline(y=0, color="k", linestyle="--", lw = 1)
    plt.xlabel("Method")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    df = load_results(args.indir, args.pattern, args.infiles)

    print("\n==== Summary ====\n", summarize(df))

    pairs = [("NCO","Markowitz"), ("Posterior-NCO","Markowitz"), ("Posterior-NCO","NCO")]
    print("\n==== Head-to-Head Win Rates ====\n", head_to_head(df, pairs))

    plot_sharpe_box(df)


if __name__ == "__main__":
    main()
