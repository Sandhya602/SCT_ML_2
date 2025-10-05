"""load_data.py
Utilities to load and preprocess customer dataset.
"""
from typing import Tuple, List
import pandas as pd
import numpy as np


DEFAULT_FEATURES = [
"AnnualIncome",
"SpendingScore",
"TotalPurchases",
"AvgPurchaseValue",
"Recency",
]




def load_csv(path: str) -> pd.DataFrame:
df = pd.read_csv(path)
return df




def validate_columns(df: pd.DataFrame, required: List[str]):
missing = [c for c in required if c not in df.columns]
if missing:
raise ValueError(f"Missing required columns: {missing}")




def preprocess(df: pd.DataFrame, features: List[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
"""Return (X, original_df) where X is numpy array of selected features.


- Fills missing numeric values with median
- Drops rows missing all features
"""
features = features or DEFAULT_FEATURES
validate_columns(df, features)
df_clean = df.copy()


# keep only relevant columns plus CustomerID if present
cols = [c for c in df_clean.columns if c in features or c == "CustomerID"]
df_clean = df_clean[cols]


# fill numeric missing with median
for f in features:
if df_clean[f].isna().any():
df_clean[f] = df_clean[f].fillna(df_clean[f].median())


X = df_clean[features].astype(float).values
return X, df_clean




if __name__ == "__main__":
import argparse
p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
args = p.parse_args()
df = load_csv(args.data)
X, df_clean = preprocess(df)
print(f"Loaded {len(X)} samples with {X.shape[1]} features")
