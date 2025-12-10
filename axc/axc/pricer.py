import pandas as pd
import yfinance as yf
from icecream import ic

class Pricer:
    def __init__(self):
        # Initialize any required attributes
        pass

    def load_market_data(self):
        """
        Load monthly returns for S&P 500, Russell 2000, and gold prices
        """
        # Load S&P 500 data
        return {x: self.get_returns(x) for x in [ '^GSPC', '^RUT', 'GC=F' ] }

    def get_returns(self, index):
            df = yf.download(index, start="2000-01-01", interval="1mo")
            df.index.name = 'Date'
            df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            return df['Close'].pct_change().dropna()


