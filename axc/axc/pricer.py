#!/usr/bin/env python3
"""
SPDX-License-Identifier: MIT

Pricing model for financial products
"""

import pandas as pd
import yfinance as yf
from icecream import ic
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Pricer:
    """
    Pricing module for financial products
    """

    def __init__(self) -> None:
        """
        set constructor
        """
        self.return_names = ["^GSPC", "^RUT", "GC=F", "^990100-USD-STRD"]
        self.scaler = StandardScaler()
        self.random_state = 42

    def get_returns(self) -> pd.DataFrame:
        """
        Load monthly returns for S&P 500, Russell 2000, and gold prices
        """
        return pd.concat({x: self.get_return(x) for x in self.return_names}, axis=1)

    def get_return(self, index: str) -> pd.Series:
        """
        get returns
        """
        df = yf.download(index, start="2000-01-01", interval="1mo")
        df.index.name = "Date"
        df.columns = ["Close", "High", "Low", "Open", "Volume"]
        return df["Close"].pct_change().dropna()

    def get_factors(
        self, returns_in: pd.DataFrame, n: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        get factors
        """
        returns = returns_in.dropna()
        factor_names = [f"Factor{i}" for i in range(1, n + 1)]
        returns_scaled = self.scaler.fit_transform(returns)
        fa = FactorAnalysis(n_components=n, random_state=self.random_state)
        factors = fa.fit_transform(returns_scaled)
        loadings = fa.components_.T
        return (
            pd.DataFrame(factors, columns=factor_names, index=returns.index),
            pd.DataFrame(loadings, columns=factor_names, index=returns.columns),
        )


def plot_factors(df: pd.DataFrame) -> None:
    """
    Plot scatter
    """
    df.plot()
    plt.show()


def plot_scatter_3d(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Plot scatter
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], cmap="viridis")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    plt.show()


if __name__ == "main":
    pricer = Pricer()
    ic(pricer.load_market_data())
