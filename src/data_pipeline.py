"""
data_pipeline.py
Author: Juan Carlos Garcia
Purpose: Automated data pipeline for downloading and saving financial time-series data.
"""

import argparse
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def download_data(tickers, start="2015-01-01", end=None, output_dir="../data"):
    """
    Downloads OHLCV data for multiple tickers using yfinance.
    Saves each to CSV in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading data for: {', '.join(tickers)}")

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                print(f"⚠️ No data found for {ticker}")
                continue
            file_path = os.path.join(output_dir, f"{ticker}.csv")
            data.to_csv(file_path)
            print(f"✅ Saved {ticker} data to {file_path}")
        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}")


def plot_data(ticker, data_dir="../data"):
    """
    Simple visualization for quick validation.
    """
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"File not found for {ticker}")
        return

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    df["Close"].plot(title=f"{ticker} Closing Price", figsize=(10, 4))
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV data using yfinance.")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers to download.")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()

    download_data(args.tickers, args.start, args.end)
