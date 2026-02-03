import pandas as pd
from pathlib import Path

FEATURES_PATH = Path("data/features/NIFTY50_features_enriched.csv")
FINAL_PATH = Path("data/features/NIFTY50_final.csv")

HORIZON = 5         
THRESHOLD = 0.008   


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    
    df["future_return"] = (
        df["close"].shift(-HORIZON) / df["close"] - 1
    )

   
    df["target"] = (df["future_return"] >= THRESHOLD).astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = create_target(df)

   
    df = df.dropna().reset_index(drop=True)

    df.to_csv(FINAL_PATH, index=False)
    print("Saved final ML-ready dataset → data/features/NIFTY50_final.csv")
