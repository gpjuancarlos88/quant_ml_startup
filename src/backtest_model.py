import sys, os
sys.path.append(os.path.abspath("../src"))  # one level up, then into src

from train_model import get_trained_models
import pandas as pd
import numpy as np
import os

def backtest(model,X_test,y_test, dates):
    y_pred = model.predict(X_test)
    signal = np.where(y_pred > 0, 1, 0)
    strat_ret = y_test.values * signal

    cum_ret = (1 + strat_ret).cumprod() - 1
    sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std()
    hit_rate = (np.sign(y_pred) == np.sign(y_test.values)).mean()

    return {"Sharpe": sharpe, "HitRate": hit_rate, "TotalReturn": cum_ret[-1]}

def main():
    models, (X_train, X_test, y_train, y_test, dates) = get_trained_models("SPY")

    print(f"Backtesting models...")
    results = {}
    for name, model in models.items():
        stats = backtest(model,X_test, y_test,dates)
        results[name] = stats
        print(f"{name:15s} | Sharpe={stats['Sharpe']:.2f} | Hit Rate={stats['HitRate']:.2%} | Total Return={stats['TotalReturn']*100:.2f}%")

    #Save summary stats
    df_results = pd.DataFrame(results).T
    df_results.to_csv("../reports/SPY_backtest_summary.csv")
    print("\n✅ Saved results to ../reports/SPY_backtest_summary.csv")

if __name__ == "__main__":
    main()

