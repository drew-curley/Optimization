import numpy as np
import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PATH = r"C:\Users\dcurl\Desktop\Input\mcmc"
FILENAME = "donors.csv"
N_DONORS = 10000  # Change this to generate more donors

# Set seed for reproducibility
np.random.seed(42)


# ============================================================
# DATA GENERATION FUNCTIONS
# ============================================================

def generate_donor_data(n_donors):
    """Generate realistic synthetic donor data."""

    data = {}

    # Age: 25-95, skewed toward older (donors tend to be older)
    data['age'] = np.clip(
        np.random.normal(loc=55, scale=15, size=n_donors).astype(int),
        25, 95
    )

    # Gender: roughly 50/50
    data['gender'] = np.random.choice(
        ['m', 'f'], size=n_donors, p=[0.48, 0.52])

    # Marital status: y=married, n=single, w=widowed
    # Widowed probability increases with age
    marital = []
    for age in data['age']:
        if age > 70:
            status = np.random.choice(['y', 'n', 'w'], p=[0.45, 0.15, 0.40])
        elif age > 50:
            status = np.random.choice(['y', 'n', 'w'], p=[0.60, 0.25, 0.15])
        else:
            status = np.random.choice(['y', 'n', 'w'], p=[0.50, 0.45, 0.05])
        marital.append(status)
    data['marital status'] = marital

    # Region: 1-10
    data['region'] = np.random.randint(1, 11, size=n_donors)

    # Tenure: months as donor (6 months to 15 years)
    data['tenure'] = np.random.randint(6, 180, size=n_donors)

    # Capacity score: 0-100 (wealth indicator)
    # Use beta distribution for realistic spread
    data['capacity score'] = np.clip(
        np.random.beta(2, 5, size=n_donors) * 100 +
        np.random.normal(20, 10, size=n_donors),
        0, 100
    ).astype(int)

    # Frequency: gifts per month (most donors give 0.05-1 times per month)
    # Higher capacity donors tend to give more frequently
    base_freq = 0.05 + (data['capacity score'] / 100) * 0.3
    data['frequency'] = np.clip(
        base_freq + np.random.exponential(0.1, size=n_donors),
        0.01, 2.0
    ).round(2)

    # Recency: months since last gift (0-36)
    # Active donors have lower recency
    data['recency'] = np.clip(
        np.random.exponential(8, size=n_donors),
        0, 36
    ).astype(int)

    # Last gift amount: $0 to $10,000,000
    # Correlated with capacity score, but with high variance
    # Some donors (lapsed) have $0
    gift_amounts = []
    for i in range(n_donors):
        capacity = data['capacity score'][i]
        recency = data['recency'][i]

        # Probability of $0 gift increases with recency
        p_zero = min(0.4, recency / 50)

        if np.random.random() < p_zero:
            amount = 0
        else:
            # Log-normal distribution scaled by capacity
            log_mean = 6 + (capacity / 100) * 4  # $400 to $22,000 base
            log_std = 1.5
            amount = np.random.lognormal(log_mean, log_std)

            # Occasional mega-donors (capacity > 80)
            if capacity > 80 and np.random.random() < 0.1:
                amount *= np.random.uniform(10, 100)

            # Cap at $10 million
            amount = min(amount, 10_000_000)

        gift_amounts.append(round(amount, 2))

    data['last gift amount'] = gift_amounts

    return pd.DataFrame(data)


def add_realistic_patterns(df):
    """Add realistic correlations and patterns to the data."""

    # Wealthy regions (1, 2, 8) have higher capacity scores
    wealthy_regions = [1, 2, 8]
    mask = df['region'].isin(wealthy_regions)
    df.loc[mask, 'capacity score'] = np.clip(
        df.loc[mask, 'capacity score'] + 15, 0, 100
    )

    # Longer tenure donors have slightly higher gift amounts
    tenure_bonus = (df['tenure'] / 180) * 0.2
    non_zero_mask = df['last gift amount'] > 0
    df.loc[non_zero_mask,
           'last gift amount'] *= (1 + tenure_bonus[non_zero_mask])
    df['last gift amount'] = df['last gift amount'].round(2)

    # Reorder columns to match expected format
    column_order = [
        'recency', 'frequency', 'tenure', 'capacity score',
        'last gift amount', 'region', 'age', 'gender', 'marital status'
    ]

    return df[column_order]


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print(f"Generating {N_DONORS} synthetic donor records...")

    # Generate data
    df = generate_donor_data(N_DONORS)
    df = add_realistic_patterns(df)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Save to CSV
    full_path = os.path.join(OUTPUT_PATH, FILENAME)
    df.to_csv(full_path, index=False)

    print(f"\nData saved to: {full_path}")
    print(f"\nData Summary:")
    print(f"  Total donors: {len(df)}")
    print(f"  Donors with gifts: {(df['last gift amount'] > 0).sum()}")
    print(
        f"  Average gift (non-zero): ${df[df['last gift amount'] > 0]['last gift amount'].mean():,.2f}")
    print(f"  Max gift: ${df['last gift amount'].max():,.2f}")
    print(
        f"  Capacity score range: {df['capacity score'].min()} - {df['capacity score'].max()}")

    print(f"\nPreview:")
    print(df.head(10).to_string())

    # Also print column info
    print(f"\nColumn Types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
