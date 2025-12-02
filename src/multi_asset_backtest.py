"""
multi_asset_backtest.py
Author: Juan Carlos Garcia Prieto
Purpose: Compare ML performance across different tickers
"""

import pandas as pd
import os
from train_model import get_trained_models
from backtest_model import backtest


TICKERS = ['SPY',"QQQ", "IWM"]

def run_multi_asset_backtest(tickers=TICKERS):
    summary = []

    for ticker in tickers:
        print(f"\n backtesting models for {ticker}... \n")

        #ensure data exists
        feature_path = f"../data/features/{ticker}_features.csv"
        if not os.path.exists(feature_path):
            print(f"Missing features for {ticker}. Generate them first")
            continue

        models, (X_train, X_test, y_train, y_test, dates) = get_trained_models(ticker)

        for name, model, in models.items():
            stats = backtest(model, X_test, y_test, dates)
            summary.append({
                "Ticker": ticker,
                "Model": name,
                **stats
            })
    
    df_sumamry = pd.DataFrame(summary)
    os.makedirs("../reports", exist_ok=True)
    df_sumamry.to_csv("../reports/multi_asset_summary.csv", index=False)
    print(f"Saved results to: ../reports/multi_asset_summary.csv")
    return df_sumamry

if __name__ == "__main__":
    df_sumamry = run_multi_asset_backtest()
    print(f"Top performing models by sharpe")
    print(df_sumamry.sort_values("Sharpe", ascending=False).head(5))

