
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/features/NIFTY50_features.csv")
OUTPUT_PATH = Path("data/features/NIFTY50_features_enriched.csv")


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)

    # Rolling volatility (std of returns)
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = add_rolling_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved enriched features → data/features/NIFTY50_features_enriched.csv")
