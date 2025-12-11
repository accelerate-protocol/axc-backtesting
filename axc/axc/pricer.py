#!/usr/bin/env python3
"""
SPDX-License-Identifier: MIT

Pricing model for financial products
"""

import pandas as pd
import yfinance as yf
from icecream import ic


class Pricer:
    """
    Pricing module for financial products
    """
    def __init__(self) -> None:
        """
        set constructor
        """
        self.returns = ["^GSPC", "^RUT", "GC=F"]

    def load_market_data(self) -> pd.DataFrame:
        """
        Load monthly returns for S&P 500, Russell 2000, and gold prices
        """
        return pd.concat(
            {x: self.get_returns(x) for x in self.returns}, axis=1
        ).dropna()

    def get_returns(self, index: str) -> pd.Series:
        """
        get returns
        """
        df = yf.download(index, start="2000-01-01", interval="1mo")
        df.index.name = "Date"
        df.columns = ["Close", "High", "Low", "Open", "Volume"]
        return df["Close"].pct_change().dropna()

if __name__ == "main":
    pricer = Pricer()
    ic(pricer.load_market_data())
