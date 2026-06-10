import pandas as pd
import numpy as np

def make_dataset(n, seed, fname):
    """Generate a simple synthetic dataset.
    Columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
    """
    np.random.seed(seed)
    ids = [f"CUST_{i:06d}" for i in range(1, n + 1)]
    ages = np.random.randint(18, 70, n)
    genders = np.random.choice(['Male', 'Female'], n, p=[0.52, 0.48])
    # Income in dollars, then convert to k$
    incomes = np.random.normal(60000, 20000, n).clip(15000, 150000).astype(int)
    spending = np.random.randint(1, 101, n)  # 1‑100 inclusive
    df = pd.DataFrame({
        'CustomerID': ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': (incomes // 1000),
        'Spending Score (1-100)': spending
    })
    df.to_csv(fname, index=False)
    print(f"Created {fname} – {n} rows (seed={seed})")

def make_mixed_dataset(n, seed, fname):
    """Generate a dataset with two distinct customer groups.
    Low‑income/low‑spending vs high‑income/high‑spending.
    """
    np.random.seed(seed)
    half = n // 2
    ids = [f"CUST_{i:06d}" for i in range(1, n + 1)]
    ages = np.random.randint(18, 70, n)
    genders = np.random.choice(['Male', 'Female'], n, p=[0.5, 0.5])

    # Low‑income group (≈ $30k‑$50k)
    low_income = np.random.normal(40000, 5000, half).clip(15000, 80000)
    # High‑income group (≈ $80k‑$120k)
    high_income = np.random.normal(90000, 5000, n - half).clip(80000, 150000)
    incomes = np.concatenate([low_income, high_income])

    # Spending scores: low group 1‑34, high group 66‑100
    low_spending = np.random.randint(1, 35, half)
    high_spending = np.random.randint(66, 101, n - half)
    spending = np.concatenate([low_spending, high_spending])

    # Shuffle rows to mix the two groups
    perm = np.random.permutation(n)
    df = pd.DataFrame({
        'CustomerID': np.array(ids)[perm],
        'Gender': genders[perm],
        'Age': ages[perm],
        'Annual Income (k$)': (incomes[perm] // 1000),
        'Spending Score (1-100)': spending[perm]
    })
    df.to_csv(fname, index=False)
    print(f"Created {fname} – {n} rows mixed groups (seed={seed})")

if __name__ == "__main__":
    # Small dataset (100 rows)
    make_dataset(n=100, seed=42, fname="sample_customers_small.csv")
    # Mixed‑group dataset (500 rows)
    make_mixed_dataset(n=500, seed=1234, fname="sample_customers_mixed.csv")
    # Large dataset (2000 rows)
    make_dataset(n=2000, seed=2022, fname="sample_customers_large.csv")
