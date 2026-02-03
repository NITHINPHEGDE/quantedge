import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_PATH = Path("data/processed/NIFTY50.csv")
FEATURES_PATH = Path("data/features")
FEATURES_PATH.mkdir(parents=True, exist_ok=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2)

    upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
    middle_col = [c for c in bb.columns if c.startswith("BBM")][0]
    lower_col = [c for c in bb.columns if c.startswith("BBL")][0]

    df["bb_upper"] = bb[upper_col]
    df["bb_middle"] = bb[middle_col]
    df["bb_lower"] = bb[lower_col]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = add_indicators(df)

    df.to_csv(FEATURES_PATH / "NIFTY50_features.csv", index=False)
    print("Saved features to data/features/AAPL_features.csv")
