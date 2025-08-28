import numpy as np
import pandas as pd

def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown of a return series (based on compounded wealth).
    """
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())

def evaluate_portfolio(weights: pd.Series, returns: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Evaluate mean, volatility, Sharpe ratio, max drawdown, and HHI of a fixed-weight portfolio.
    """
    w = weights.reindex(returns.columns, fill_value=0.0).astype(float)
    if not np.isclose(w.sum(), 1.0):
        s = w.sum()
        if s <= 0:
            w[:] = 1.0 / len(w)
        else:
            w = w / s

    portfolio_returns = returns @ w  # 이제 정렬 완료라 안전
    mean_return = float(portfolio_returns.mean())
    volatility = float(portfolio_returns.std(ddof=1))
    sharpe = (mean_return - risk_free_rate) / (volatility + 1e-12)
    mdd = max_drawdown(portfolio_returns)
    hhi = float((w ** 2).sum())
    return {
        "mean": mean_return,
        "vol": volatility,
        "sharpe": sharpe,
        "mdd": float(mdd),
        "hhi": hhi
    }