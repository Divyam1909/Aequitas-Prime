"""
Download the UCI Adult Income dataset into data/raw/.
Uses sklearn's fetch_openml (OpenML mirror) — reliable even when UCI is down.
Run from project root: python scripts/download_data.py
"""

from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "adult.csv"

    if dest.exists():
        print(f"  adult.csv already exists at {dest}, skipping download.")
        return

    print("  Fetching Adult Income dataset via OpenML (sklearn)...")
    dataset = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = dataset.frame

    # Rename the target column to 'income' for consistency
    df = df.rename(columns={"class": "income"})

    df.to_csv(dest, index=False)
    print(f"  Saved {len(df)} rows to {dest}")
    print(f"  Columns: {list(df.columns)}")
    print("\nDone. Dataset ready in data/raw/adult.csv")

if __name__ == "__main__":
    download()
