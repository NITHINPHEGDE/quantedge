import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR=Path("data/raw")


def fetch_ohlcv(ticker,start,end):
  df=yf.download(ticker,start=start,end=end,progress=False)
  df.reset_index(inplace=True)
  df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
        for c in df.columns]
  df.rename(columns={"date": "timestamp"}, inplace=True)
  return df


def save_raw_data(df,ticker):
  RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
  file_path = RAW_DATA_DIR / f"{ticker}.csv"
  df.to_csv(file_path, index=False)
  print(f"Saved raw data → {file_path}")


if __name__ == "__main__":
  TICKER = "^NSEI"
TICKER_NAME = "NIFTY50"

df = fetch_ohlcv(
    ticker=TICKER,
    start="2018-01-01",
    end="2025-01-01"
)
save_raw_data(df, TICKER_NAME)
