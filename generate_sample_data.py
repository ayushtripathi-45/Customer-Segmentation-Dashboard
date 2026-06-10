import pandas as pd
import numpy as np

np.random.seed(42)

n_customers = 500

customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
ages = np.random.randint(18, 70, n_customers)
annual_incomes = np.random.normal(60000, 20000, n_customers).clip(15000, 150000).astype(int)
spending_scores = np.random.randint(1, 100, n_customers)

gender = np.random.choice(['Male', 'Female'], n_customers, p=[0.52, 0.48])

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': gender,
    'Age': ages,
    'Annual Income (k$)': (annual_incomes / 1000).astype(int),
    'Spending Score (1-100)': spending_scores
})

df.to_csv('sample_customers.csv', index=False)
print("Sample dataset created: sample_customers.csv")
print(f"Shape: {df.shape}")
print(df.head())
print("\nBasic Statistics:")
print(df.describe())