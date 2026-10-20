import sys
import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timezone


def _next_month(y: int, m: int):
    m += 1
    if m > 12:
        m = 1
        y += 1
    return y, m


class DataCalculator:
    DEFAULT_CSV_PATH = "/workspace/yearly_data_final.csv"

    def __init__(self, filename: str = None):
        self.filename = filename if filename is not None else self.DEFAULT_CSV_PATH
        df = pd.read_csv(self.filename)
        # Convert Return to float, stripping non-numeric chars like '*'
        df["Return"] = df["Return"].astype(str).str.replace(r"[^0-9.-]", "", regex=True).astype(float)
        self.df = df

    def calculate_cumulative_return(self, target_month: int, target_year: int, price: float, start_year: int = None, start_month: int = None, unix_timestamp: bool = False) -> pd.DataFrame:
        """
        Calculate cumulative return series from start_month/start_year to end of data.

        The price is the price at the START of target_month/target_year.
        - For months after target: propagate forward (multiply by 1 + return)
        - For months before target: propagate backward (divide by 1 + return)
        - Series spans from start_month/start_year to end of data

        Args:
            target_month: Target month (1-12) where price is known
            target_year: Target year where price is known
            price: Price at START of target_month/target_year
            start_year: Series start year (defaults to target_year)
            start_month: Series start month (defaults to target_month)
            unix_timestamp: If True, return unix_timestamp (end of month, UTC)

        Returns:
            DataFrame with columns: Month, Year, Cumulative_Return
            (or Cumulative_Return, unix_timestamp when unix_timestamp=True)
        """
        # Default start to target if not specified
        if start_year is None:
            start_year = target_year
        if start_month is None:
            start_month = target_month

        # Build full month list from start to end of data
        all_months = [(int(r["Year"]), int(r["Month"])) for _, r in self.df.iterrows()]
        all_returns = { (int(r["Year"]), int(r["Month"])): r["Return"] for _, r in self.df.iterrows() }

        # Filter months from start_year/start_month onward
        months = [(y, m) for y, m in all_months if (y > start_year) or (y == start_year and m >= start_month)]
        if not months:
            return pd.DataFrame(columns=["Month", "Year", "Cumulative_Return"])

        # Find target index in months list
        target_idx = None
        for i, (y, m) in enumerate(months):
            if y == target_year and m == target_month:
                target_idx = i
                break

        if target_idx is None:
            return pd.DataFrame(columns=["Month", "Year", "Cumulative_Return"])

        # Build cumulative returns: start at target with price, go forward and backward.
        # cumulative[i] = price at the START of month months[i].
        # The return for a given month is realized at the END of that month.
        cumulative = [0.0] * len(months)
        cumulative[target_idx-1] = price  # price at START of target_month/target_year

        # Propagate forward from target:
        #   start-of-month price for month i =
        #     start-of-month price for month i-1 * (1 + return of month i-1)
        # The return of month i-1 is realized at the END of month i-1,
        # which equals the START of month i — so this is the correct link.
        for i in range(target_idx, len(months)):
            y, m = months[i]          # return of the PREVIOUS month
            r = all_returns[(y, m)]       # realized at END of month (i-1)
            cumulative[i] = cumulative[i - 1] * (1 + r / 100)

        # Propagate backward from target:
        #   start-of-month price for month i =
        #     start-of-month price for month i+1 / (1 + return of month i)
        # The return of month i is realized at the END of month i,
        # which equals the START of month i+1 — so dividing by
        # (1 + return of month i) unwinds one month of compounding.
        for i in range(target_idx-1, 0, -1):
            y, m = months[i]              # return of the CURRENT month
            r = all_returns[(y, m)]       # realized at END of month i
            cumulative[i-1] = cumulative[i] / (1 + r / 100)

        if not unix_timestamp:
            return pd.DataFrame({
                "Month": [m for _, m in months],
                "Year": [y for y, _ in months],
                "Cumulative_Return": cumulative,
            })

        stamps = []
        for y, m in months:
            days = calendar.monthrange(y, m)[1]
            stamps.append(int(datetime(y, m, days, 23, 59, 59, tzinfo=timezone.utc).timestamp()))
        return pd.DataFrame({"Cumulative_Return": cumulative, "unix_timestamp": stamps})

    def calculate_price(self, start_year: int, start_month: int, start_price: float,
                        target_year: int, target_month: int, target_day: int) -> float:
        """
        Calculate the price at a target date, compounding monthly returns.
        
        Args:
            start_year: Starting target_year
            start_month: Starting month (1-12)
            start_price: Initial price
            target_year: Target year
            target_month: Target month (1-12)
            target_day: Target day (1-31)
        
        Returns:
            Price at target date
        """
        # Find start index
        start_idx = None
        for i, row in self.df.iterrows():
            if row["Year"] == start_year and row["Month"] == start_month:
                start_idx = i
                break
        
        if start_idx is None:
            raise ValueError(f"Start date {start_year}-{start_month:02d} not found in data")
        
        # Get data from start month onward
        data_rows = self.df.loc[start_idx:].copy().reset_index(drop=True)
        
        # Calculate total months from start to target
        total_months = (target_year - start_year) * 12 + (target_month - start_month)
        
        price = start_price
        last_return = None
        
        for m in range(total_months + 1):  # Include target month
            cy = start_year + (start_month - 1 + m) // 12
            cm = (start_month - 1 + m) % 12 + 1
            
            # Find return for this year/month
            row = data_rows[(data_rows["Year"] == cy) & (data_rows["Month"] == cm)]
            if len(row) > 0:
                r = row.iloc[0]["Return"]
                last_return = r
            else:
                # Past the end of data - use last known return
                if last_return is None:
                    # Shouldn't happen if start is in data
                    r = 0.0
                else:
                    r = last_return
            
            if m < total_months:
                # Full month compound
                price *= (1 + r / 100)
            else:
                # Target month - interpolate based on day
                days_in_month = calendar.monthrange(cy, cm)[1]
                fraction = target_day / days_in_month
                price *= (1 + (r / 100) * fraction)
        
        return price

    def estimated_apy(self, year: int, month: int) -> float:
        """
        Estimate the annual percentage yield (APY) for the given year/month.

        Compounds the monthly returns from (year-1, month+1) through
        (year, month) — a 12-month lookback window — and annualizes the
        result to a percentage.

        Args:
            year: Year (e.g., 2020)
            month: Month (1-12)

        Returns:
            Estimated APY as a percentage (e.g., 15.0 means 15.0%)
        """
        # Start month rolls into the next year when month == 12
        start_month_raw = month + 1
        start_year = year - 1 + (start_month_raw - 1) // 12
        start_month = (start_month_raw - 1) % 12 + 1

        # Vectorized 12-month lookback filter
        df = self.df.copy()
        df["ym"] = df["Year"] * 100 + df["Month"]

        start_ym = start_year * 100 + start_month
        end_ym = year * 100 + month

        subset = df[(df["ym"] >= start_ym) & (df["ym"] <= end_ym)]
        if subset.empty:
            raise ValueError(f"No return data found for window ending {year}-{month:02d}")
        print(subset)
        # Compound monthly returns and annualize
        monthly_factors = 1 + subset["Return"] / 100
        product = monthly_factors.prod()
        n_months = len(subset)
        apy = (product ** (12 / n_months) - 1) * 100
        return apy

    @property
    def data(self) -> pd.DataFrame:
        return self.df


if __name__ == "__main__":
    # Use command-line argument for filename if provided, else default
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    calculator = DataCalculator(filename)
    print(calculator.data)
    print(f"\nShape: {calculator.data.shape}")
    print(f"Columns: {list(calculator.data.columns)}")
    print(f"Dtypes:\n{calculator.data.dtypes}")

    # Demonstrate cumulative return calculation
    print("\n--- Cumulative Return Demonstration ---")
    # Example: start from Jan 2020 with initial price 100
    cum_df = calculator.calculate_cumulative_return(target_month=1, target_year=2020, price=100.0)
    print("Cumulative return starting Jan 2020 with start_price=100:")
    print(cum_df)

    print("\n--- Cumulative Return with Unix Timestamps Demonstration ---")
    # Use unix_timestamp flag
    ts_df = calculator.calculate_cumulative_return(target_month=1, target_year=2020, price=100.0, unix_timestamp=True)
    print("Cumulative return with unix_timestamp (end of month, return first):")
    print(ts_df)
    
    # Demonstrate price calculation
    print("\n--- Price Calculation Demonstration ---")
    # Example: start from Jan 2020 with initial price 100, calculate price for March 15, 2020
    target_price = calculator.calculate_price(
        start_year=2020, 
        start_month=1, 
        start_price=100.0,
        target_year=2020, 
        target_month=3, 
        target_day=15
    )
    print(f"Price on March 15, 2020 with start_price=100: {target_price:.2f}")
    # Demonstrate estimated APY calculation
    print("\n--- Estimated APY Demonstration ---")
    apy_dec2020 = calculator.estimated_apy(year=2020, month=12)
    print(f"Estimated APY for December 2020: {apy_dec2020:.2f}%")
    apy_jun2021 = calculator.estimated_apy(year=2021, month=6)
    print(f"Estimated APY for June 2021: {apy_jun2021:.2f}%")

    apy_jan2022 = calculator.estimated_apy(year=2022, month=1)
    print(f"Estimated APY for January 2022: {apy_jan2022:.2f}%")
