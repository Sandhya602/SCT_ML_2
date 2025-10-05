import csv
import random
from pathlib import Path


OUT = Path('data/mall_customers.csv')
OUT.parent.mkdir(parents=True, exist_ok=True)


with open(OUT, 'w', newline='') as f:
writer = csv.writer(f)
writer.writerow(['CustomerID','Gender','Age','AnnualIncome','SpendingScore','TotalPurchases','AvgPurchaseValue','Recency'])
for i in range(1,501):
gender = random.choice(['Male','Female'])
age = random.randint(18,70)
income = random.randint(10,150)
spending = random.randint(1,100)
total_purchases = random.randint(1,200)
avg_val = round(random.uniform(10,500),2)
recency = random.randint(1,365)
writer.writerow([i, gender, age, income, spending, total_purchases, avg_val, recency])


print(f"Synthetic data written to {OUT}")
