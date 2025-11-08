"""
trian_model.py
Autor: Juan Carlos Garcia Prieto
Purpose: To train ML Models on engineered features
to predict next days stock returns and compare performance.
"""

import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score



def load_features(ticker, data_dir = "../data/features"):
    file_path = os.path.join(data_dir, f"{ticker}_features.csv")
    df = pd.read_csv(file_path,index_col=0, parse_dates= True)
    return df

def prepare_data(df):
    #Target: next days return
    df["Return"] = df["Close"].pct_change().shift(-1)
    df.dropna(inplace = True)

    #Features: All technical indicators
    feature_col = ["SMA_10", "SMA_30", "EMA_10", "EMA_30", "MOM_5", "VOL_20", "RSI_14"]
    X = df[feature_col]
    y= df["Return"]

    #Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Split chronologically
    X_train, X_test, y_trian, y_test = train_test_split(
        X_scaled, y, test_size= 0.2, shuffle= False
    )

    return X_train, X_test, y_trian, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    # -- Linear Regression --
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["Linear"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "R2": r2_score(y_test, y_pred_lr)
    }

    # -- LASSO --
    lasso = LassoCV(cv=5)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    results['LASSO'] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
        "R2": r2_score(y_test, y_pred_lasso)
    }

    # -- RandomForest --
    rf = RandomForestRegressor(n_estimators=100, max_depth= 5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf)
    }

        # --- Directional Accuracy ---
    y_pred_sign = np.sign(y_pred_rf)
    y_true_sign = np.sign(y_test)
    acc = (y_pred_sign == y_true_sign).mean()
    results["Random Forest"]["Direction_Accuracy"] = acc


    return results

def get_trained_models(ticker):
    """
    Loads features, prepares data, trains multiple models
    and returns them. Used by backtesting module.
    """
    df = load_features(ticker)
    X_train, X_test, y_train, y_test = prepare_data(df)
    results = {}

    #Train models
    lr = LinearRegression().fit(X_train, y_train)
    lasso = LassoCV().fit(X_train,y_train)
    rf = RandomForestRegressor().fit(X_train,y_train)

    results["Linear"] = lr
    results["LASSO"] = lasso
    results["RandomForest"] = rf

    return results, (X_train, X_test, y_train, y_test, df.index[-len(y_test):])


def main():
    df = load_features("SPY")
    X_train, X_test, y_train, y_test = prepare_data(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(f"Model Performance")
    for model, metrics in results.items():
         print(f"{model}: RMSE={metrics['RMSE']:.6f}, R²={metrics['R2']:.4f}")

if __name__ == "__main__":
    main()



