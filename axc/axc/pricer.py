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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

ic.enable()

class Pricer:
    """
    Pricing module for financial products
    """

    def __init__(self, return_names) -> None:
        """
        set constructor
        """
        self.return_names = return_names
        self.scaler = StandardScaler()
        self.random_state = 42

    def load_market_data(self, filename: str) -> pd.DataFrame:
        """
        Load returns
        """
        df = self.replace_last_day_of_month(pd.read_csv(filename), "Date")
        return df.sort_values(by="Date").set_index("Date")

    def get_returns(self) -> pd.DataFrame:
        """
        Load monthly returns for S&P 500, Russell 2000, and gold prices
        """
        return pd.concat(
            {x: self.get_return(x) for x in self.return_names}, axis=1
        ).dropna()

    def get_return(self, index: str) -> pd.Series:
        """
        get returns
        """
        df = yf.download(index, start="2000-01-01")
        df.index.name = "Date"
        df.columns = ["Close", "High", "Low", "Open", "Volume"]
        df = df[["Close"]]
        df["prev_month_end"] = df.index - pd.DateOffset(months=1)
        df["prev_month_close"] = (
            #            df["prev_month_end"].map(lambda x: item(df["Close"], x)).ffill().dropna()
            df["prev_month_end"]
            .map(lambda x: df["Close"].get(x, None))
            .ffill()
            .dropna()
        )
        df["return"] = (
            (df["Close"] - df["prev_month_close"]) / df["prev_month_close"] * 100
        )
        return df["return"].ffill().dropna()

    def get_factors(
        self, returns_in: pd.DataFrame, n: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        get factors
        """
        scaler = StandardScaler()
        returns = returns_in.dropna()
        returns_scaled = scaler.fit_transform(returns)
#        returns_scaled = returns
        factor_names = self.factor_names(n)
        fa = FactorAnalysis(n_components=n, random_state=self.random_state)
        factors = fa.fit_transform(returns_scaled)
        loadings = fa.components_.T
        return (
            pd.DataFrame(factors, columns=factor_names, index=returns.index),
            pd.DataFrame(loadings, columns=factor_names, index=returns.columns),
        )

    def collate_returns(self, factors: pd.DataFrame, target: pd.DataFrame):
        """
        collate returns
        """
        df = pd.concat([factors, target], axis=1)
        df.iloc[:, :-1] = df.iloc[:, :-1].ffill()
        return df[~df['Return'].isna()]

    def get_regression(self, df: pd.DataFrame):
        """
        do regression
        """
        df = df.copy()
        X = df.iloc[:, :-1]  # All columns except the last one
        y = df.iloc[:, -1]
 #       model = LinearRegression()
        model = Lasso(alpha=0.01)
        model.fit(X, y)
        y_pred = model.predict(X)
        df['predicted'] = y_pred
        ic(f"Coefficients: {model.coef_}")
        ic(f"Intercept: {model.intercept_}")
        return (model, df)

#        model = LinearRegression()
#        return model.fit(factors, target)

    @staticmethod
    def factor_names(n: int):
        return [f"Factor{i}" for i in range(1, n + 1)]

    @staticmethod
    def replace_last_day_of_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Replace the date in the specified column with the last day of the month.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the column containing dates.

        Returns:
        pd.DataFrame: A new DataFrame with the updated date column.
        """
        df[date_col] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0)
        return df


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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

ic.enable()

class Pricer:
    """
    Pricing module for financial products
    """

    def __init__(self, return_names) -> None:
        """
        set constructor
        """
        self.return_names = return_names
        self.scaler = StandardScaler()
        self.random_state = 42

    def load_market_data(self, filename: str) -> pd.DataFrame:
        """
        Load returns
        """
        df = self.replace_last_day_of_month(pd.read_csv(filename), "Date")
        return df.sort_values(by="Date").set_index("Date")

    def get_returns(self) -> pd.DataFrame:
        """
        Load monthly returns for S&P 500, Russell 2000, and gold prices
        """
        return pd.concat(
            {x: self.get_return(x) for x in self.return_names}, axis=1
        ).dropna()

    def get_return(self, index: str) -> pd.Series:
        """
        get returns
        """
        df = yf.download(index, start="2000-01-01")
        df.index.name = "Date"
        df.columns = ["Close", "High", "Low", "Open", "Volume"]
        df = df[["Close"]]
        df["prev_month_end"] = df.index - pd.DateOffset(months=1)
        df["prev_month_close"] = (
            #            df["prev_month_end"].map(lambda x: item(df["Close"], x)).ffill().dropna()
            df["prev_month_end"]
            .map(lambda x: df["Close"].get(x, None))
            .ffill()
            .dropna()
        )
        df["return"] = (
            (df["Close"] - df["prev_month_close"]) / df["prev_month_close"] * 100
        )
        return df["return"].ffill().dropna()

    def get_factors(
        self, returns_in: pd.DataFrame, n: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        get factors
        """
        scaler = StandardScaler()
        returns = returns_in.dropna()
        returns_scaled = scaler.fit_transform(returns)
#        returns_scaled = returns
        factor_names = self.factor_names(n)
        fa = FactorAnalysis(n_components=n, random_state=self.random_state)
        factors = fa.fit_transform(returns_scaled)
        loadings = fa.components_.T
        return (
            pd.DataFrame(factors, columns=factor_names, index=returns.index),
            pd.DataFrame(loadings, columns=factor_names, index=returns.columns),
        )

    def collate_returns(self, factors: pd.DataFrame, target: pd.DataFrame):
        """
        collate returns
        """
        df = pd.concat([factors, target], axis=1)
        df.iloc[:, :-1] = df.iloc[:, :-1].ffill()
        return df[~df['Return'].isna()]

    def get_regression(self, df: pd.DataFrame):
        """
        do regression
        """
        df = df.copy()
        X = df.iloc[:, :-1]  # All columns except the last one
        y = df.iloc[:, -1]
 #       model = LinearRegression()
        model = Lasso(alpha=0.01)
        model.fit(X, y)
        y_pred = model.predict(X)
        df['predicted'] = y_pred
        ic(f"Coefficients: {model.coef_}")
        ic(f"Intercept: {model.intercept_}")
        return (model, df)

#        model = LinearRegression()
#        return model.fit(factors, target)

    @staticmethod
    def factor_names(n: int):
        return [f"Factor{i}" for i in range(1, n + 1)]

    @staticmethod
    def replace_last_day_of_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Replace the date in the specified column with the last day of the month.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the column containing dates.

        Returns:
        pd.DataFrame: A new DataFrame with the updated date column.
        """
        df[date_col] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0)
        return df


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
