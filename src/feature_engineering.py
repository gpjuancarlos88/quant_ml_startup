"""
feature_engineering.py
Author: Juan Carlos Garcia
Purpose: Generate technical features from price data for quant ML models.
"""

import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    """
    Load vendor CSV exports that include extra metadata rows and ensure we
    return an OHLCV DataFrame indexed by datetime.
    """
    df = pd.read_csv(file_path)
    df_clean = df.drop([0,1])

    df_clean["SMA_10"] = df_clean["Close"].rolling(window=10).mean()
    df_clean["SMA_30"] = df_clean["Close"].rolling(window=30).mean()
    df_clean["EMA_10"] = df_clean["Close"].ewm(span=10, adjust=False).mean()
    df_clean["EMA_30"] = df_clean["Close"].ewm(span=30, adjust=False).mean()

    df_clean['Close'] = pd.to_numeric(df_clean["Close"], errors= "coerce")

     # --- Momentum ---
    df_clean["MOM_5"] = df_clean["Close"].pct_change(periods=5)

    # --- Volatility (20-day rolling standard deviation) ---
    df_clean["VOL_20"] = df_clean["Close"].pct_change().rolling(window=20).std()

    delta = df_clean["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df_clean["RSI_14"] = 100 - (100 / (1 + rs.replace(0, np.nan)))

    return df_clean



def compute_technical_indicators(df):
    """
    Takes OHLCV DataFrame and returns it with new feature columns.
    """
    df = df.copy()

    # --- Moving Averages ---
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()

    # --- Momentum ---
    df["MOM_5"] = df["Close"].pct_change(periods=5)

    # --- Volatility (20-day rolling standard deviation) ---
    df["VOL_20"] = df["Close"].pct_change().rolling(window=20).std()

    # --- RSI (Relative Strength Index, 14 periods) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs.replace(0, np.nan)))

    return df


def generate_features_for_ticker(ticker, data_dir="../data", output_dir="../data/features"):
    """
    Loads raw CSV from data_dir, computes features, and saves a new CSV in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{ticker}.csv")

    df = load_and_clean_data(file_path)

    df_feat = compute_technical_indicators(df)
    df_feat.dropna(inplace=True)

    output_path = os.path.join(output_dir, f"{ticker}_features.csv")
    df_feat.to_csv(output_path)
    print(f"✅ Saved engineered features to {output_path}")
    return df_feat