"""
This module contains functions for downloading stock data.
"""
import logging

import pandas as pd
import yfinance as yf 

logger = logging.getLogger(__name__)

def download_data(ticker:str) -> pd.DataFrame:
    """
    Downloads the stock data for the given ticker symbol and save it as CSV file.

    Args:
        ticker (str): The stock ticker symbol to download data for. Uses Yahoo Finance format.
        period (str): The period of time to download the file.
        interval (str): The interval of data points.
    Returns:
        The pandas dataframe containing the stock price of specifies period and interval.
    """
    logger.info("Downloading stock data of %s", ticker)
    try:
        df = yf.Ticker(ticker).history(period = "1y", interval = "1d")
        df = df.reset_index()
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df = df.drop(columns = ["Dividends", "Stock Splits"])
        df.to_csv(rf"data/{ticker}.csv")
        return df
    except Exception as e:
        logger.exception("Unable to download data on %s: %s", ticker, e)
        raise
