# SCT_ML_2
# Customer Segmentation with K-Means


This repo trains a K-Means clustering model to segment retail customers using purchase history and demographic features. It supports training, evaluation (elbow & silhouette), and assigning new customers to clusters.


## Dataset format


Place a CSV file at `data/mall_customers.csv`. Common columns (you can adapt to your mail/customer dataset):
Recommended features used for clustering: `AnnualIncome`, `SpendingScore`, `TotalPurchases`, `AvgPurchaseValue`, `Recency`.


## Quick start


1. Setup environment


```bash
python -m venv venv
source venv/bin/activate # macOS / Linux
venv\Scripts\activate # Windows
pip install -r requirements.txt
