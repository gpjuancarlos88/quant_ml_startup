"""
multi_asset_backtest.py
Author: Juan Carlos Garcia Prieto
Purpsoe: Compare ML performance across different ETG's
"""

import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath("../src"))
from backtest_model import backtest
from train_model import get_trained_models

Tickers = ["SPY", "QQQ","IWM"]

def run_multi_asset_backtest(ticker_=Tickers):
    summary = []

    for ticker in Tickers:
        print(f"Backtesting strategy for {ticker} ... ")
        models, (X_train, X_test, y_train, y_test, dates) = get_trained_models(ticker)

        for name, model in models.items():
            stats = backtest(model, X_test, y_test, dates)
            summary.append({
                "Ticker":ticker,
                "Model": name,
                **stats
            })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("../reports/multi_asset_summary.csv", index = False)
    print(f"Saved results to ../reports/multi_asset_summary.csv")
    return df_summary

if __name__ == "__main__":
    df_summary = run_multi_asset_backtest()
    print(f"Top performing models by Sharpe:")
    print(df_summary.sort_values("Sharpe", ascending= False).head(5))

