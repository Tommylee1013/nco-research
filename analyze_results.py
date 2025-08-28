from __future__ import annotations
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Analyze Monte Carlo results.")
    p.add_argument("--indir", type=str, default="data/synthetic")
    p.add_argument("--pattern", type=str, default="*.csv")
    return p.parse_args()

def load_results(indir: str, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {indir} matching {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], axis=0, ignore_index=True)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["shrinkage","denoising","detone","method"])[["sharpe","vol","mdd","hhi"]].agg(["mean","median"])

def head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index=["trial","shrinkage","denoising","detone"], columns="method", values="sharpe", aggfunc="mean")
    pivot = pivot.dropna(subset=["NCO","Markowitz"], how="any")
    wins = (pivot["NCO"] > pivot["Markowitz"]).groupby(level=[1,2,3]).mean().rename("nco_win_rate")
    return wins.reset_index()

def plot_sharpe_box(df: pd.DataFrame):
    ax = df.boxplot(column="sharpe", by="method")
    plt.title("Out-of-sample Sharpe by Method")
    plt.suptitle("")
    plt.xlabel("Method")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    df = load_results(args.indir, args.pattern)
    print("\n==== Summary ====\n", summarize(df))
    print("\n==== NCO vs Markowitz Win-Rate (by config) ====\n", head_to_head(df))
    plot_sharpe_box(df)

if __name__ == "__main__":
    main()
