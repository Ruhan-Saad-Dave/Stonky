import yfinance as yf 
import pandas as pd

def download_data(ticker:str, period:str = "1y", interval:str = "1d") -> pd.DataFrame:
    """
    Downloads the stock data for the given ticker symbol and save it as CSV file.

    Args:
        ticker (str): The stock ticker symbol to download data for. Uses Yahoo Finance format.
        period (str): The period of time to download the file.
        interval (str): The interval of data points.
    Returns:
        The pandas dataframe containing the stock price of specifies period and interval.
    """
    df = yf.Ticker("GOOG").history(period = "1y", interval = "1d")
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns = ["Dividends", "Stock Splits"])
    df.to_csv(rf"data/{ticker}.csv")
    return df
