import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/NIFTY50.csv")
PROCESSED_FILE = Path("data/processed/NIFTY50.csv")
PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

df.drop_duplicates(subset="timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(PROCESSED_FILE, index=False)
print("Saved processed data")
