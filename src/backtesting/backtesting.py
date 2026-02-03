import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA_PATH = Path("data/features/NIFTY50_final.csv")
MODEL_PATH = Path("models/xgboost_model.pkl")

HOLDING_PERIOD = 5
TOP_PERCENTILE = 0.90  # top 10%


def run_simple_backtest():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Prepare features
    drop_cols = ["timestamp", "target", "future_return"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict probabilities
    df["proba"] = model.predict_proba(X)[:, 1]

    # Select top 10% confidence days
    threshold = df["proba"].quantile(TOP_PERCENTILE)
    df["signal"] = df["proba"] >= threshold

    # Calculate future return
    df["trade_return"] = (
        df["close"].shift(-HOLDING_PERIOD) / df["close"] - 1
    )

    trades = df[df["signal"]].dropna()

    print("Number of trades:", len(trades))
    print("Average return per trade:",
          round(trades["trade_return"].mean() * 100, 2), "%")

    print("Cumulative return:",
          round(((1 + trades["trade_return"]).prod() - 1) * 100, 2), "%")


if __name__ == "__main__":
    run_simple_backtest()
