"""assign.py
Load trained pipeline and assign new customers to clusters.
"""
import argparse
import joblib
import json
import numpy as np
from pathlib import Path


FEATURES = [
"AnnualIncome",
"SpendingScore",
"TotalPurchases",
"AvgPurchaseValue",
"Recency",
]




def assign_from_values(model_path: str, values: list):
model = joblib.load(model_path)
X = np.array(values, dtype=float).reshape(1, -1)
labels = model.predict(X)
return int(labels[0])




def assign_from_json(model_path: str, json_path: str):
with open(json_path, 'r') as f:
data = json.load(f)
model = joblib.load(model_path)
if isinstance(data, dict):
data = [data]
results = []
for item in data:
X = [item.get(f) for f in FEATURES]
label = int(model.predict(np.array(X, dtype=float).reshape(1, -1))[0])
results.append({"input": item, "cluster": label})
return results




if __name__ == "__main__":
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--input_json")
p.add_argument("--AnnualIncome", type=float)
p.add_argument("--SpendingScore", type=float)
p.add_argument("--TotalPurchases", type=float)
p.add_argument("--AvgPurchaseValue", type=float)
p.add_argument("--Recency", type=float)
args = p.parse_args()


if args.input_json:
out = assign_from_json(args.model, args.input_json)
print(json.dumps(out, indent=2))
else:
vals = [args.AnnualIncome, args.SpendingScore, args.TotalPurchases, args.AvgPurchaseValue, args.Recency]
if any(v is None for v in vals):
p.error('Provide either --input_json or all feature values')
label = assign_from_values(args.model, vals)
print(f"Assigned to cluster: {label}")
