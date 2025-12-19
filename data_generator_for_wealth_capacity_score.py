"""
Generate Stochastic Donor Data with Real Geographic Reference Values
Automatically fetches ALL US ZIP/County/State medians from Census API (FREE)
"""

import numpy as np
import pandas as pd
import requests
import time

# ============================================================
# CONFIGURATION
# ============================================================

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_DONORS = 10000
OUTPUT_FILE = "donor_wealth_capacity_data.csv"

# Get free Census API key at: https://api.census.gov/data/key_signup.html
CENSUS_API_KEY = None  # Replace with your key or leave as None

# ============================================================
# STEP 1: FETCH ALL GEOGRAPHIC DATA FROM CENSUS API
# ============================================================

def fetch_state_medians():
    """
    Fetch median home values for ALL US states from Census API
    Uses ACS 5-Year estimates (most reliable)
    """
    print("Fetching state median home values from Census API...")
    
    url = f"https://api.census.gov/data/2021/acs/acs5"
    params = {
        "get": "B25077_001E,NAME",
        "for": "state:*"
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    headers = data[0]
    rows = data[1:]
    
    state_medians = {}
    for row in rows:
        value = row[0]
        name = row[1]
        state_code = row[2]
        
        if value and value not in ['-666666666', 'null']:
            state_medians[state_code] = float(value)
    
    if len(state_medians) == 0:
        raise ValueError("No state data retrieved from Census API")
    
    print(f"  ✓ Fetched {len(state_medians)} states")
    return state_medians

def fetch_county_medians(max_counties=500):
    """
    Fetch median home values for counties from Census API
    """
    print(f"Fetching county median home values (top {max_counties})...")
    
    url = f"https://api.census.gov/data/2021/acs/acs5"
    params = {
        "get": "B25077_001E,NAME",
        "for": "county:*"
    }
    
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    
    headers = data[0]
    rows = data[1:]
    
    county_data = []
    for row in rows:
        value = row[0]
        name = row[1]
        state_code = row[2]
        county_code = row[3]
        
        if value and value not in ['-666666666', 'null']:
            county_data.append({
                'name': name,
                'state': state_code,
                'county': county_code,
                'median': float(value)
            })
    
    if len(county_data) == 0:
        raise ValueError("No county data retrieved from Census API")
    
    county_data.sort(key=lambda x: x['median'], reverse=True)
    county_data = county_data[:max_counties]
    
    county_medians = {c['name']: c['median'] for c in county_data}
    
    print(f"  ✓ Fetched {len(county_medians)} counties")
    return county_medians

def fetch_zip_medians(sample_size=1000):
    """
    Fetch median home values for ZIP codes from Census API
    """
    print(f"Fetching ZIP code median home values (sample of {sample_size})...")
    
    url = f"https://api.census.gov/data/2021/acs/acs5"
    params = {
        "get": "B25077_001E,NAME",
        "for": "zip code tabulation area:*"
    }
    
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    
    headers = data[0]
    rows = data[1:]
    
    zip_data = []
    for row in rows:
        value = row[0]
        name = row[1]
        zcta = row[2]
        
        if value and value not in ['-666666666', 'null']:
            zip_data.append({
                'zip': zcta,
                'median': float(value)
            })
    
    if len(zip_data) == 0:
        raise ValueError("No ZIP code data retrieved from Census API")
    
    if len(zip_data) > sample_size:
        zip_data = np.random.choice(zip_data, size=sample_size, replace=False).tolist()
    
    zip_medians = {z['zip']: z['median'] for z in zip_data}
    
    print(f"  ✓ Fetched {len(zip_medians)} ZIP codes")
    return zip_medians

# ============================================================
# STEP 2: ASSIGN GEOGRAPHIC LOCATIONS
# ============================================================

def assign_geographies(n_donors, state_medians, zip_medians, county_medians):
    """
    Randomly assign each donor to a state, county, and ZIP code
    """
    print("\nAssigning geographic locations to donors...")
    
    states = list(state_medians.keys())
    zips = list(zip_medians.keys())
    counties = list(county_medians.keys())
    
    donor_states = np.random.choice(states, size=n_donors)
    donor_zips = np.random.choice(zips, size=n_donors)
    donor_counties = np.random.choice(counties, size=n_donors)
    
    state_median_values = np.array([state_medians[s] for s in donor_states])
    zip_median_values = np.array([zip_medians[z] for z in donor_zips])
    county_median_values = np.array([county_medians[c] for c in donor_counties])
    
    print(f"  States: {len(set(donor_states))} unique")
    print(f"  ZIPs: {len(set(donor_zips))} unique")
    print(f"  Counties: {len(set(donor_counties))} unique")
    
    return (donor_states, donor_zips, donor_counties, 
            state_median_values, zip_median_values, county_median_values)

# ============================================================
# STEP 3: GENERATE HOME VALUES
# ============================================================

def generate_home_values(n_donors, zip_median_values, cap=10_000_000):
    """
    Generate realistic home values centered around ZIP code medians
    """
    print("\nGenerating home values...")
    
    home_values = np.zeros(n_donors)
    
    for i in range(n_donors):
        zip_median = zip_median_values[i]
        mu = np.log(zip_median)
        sigma = 0.6
        value = np.random.lognormal(mean=mu, sigma=sigma)
        value = np.clip(value, 50000, cap)
        home_values[i] = value
    
    print(f"  Mean: ${home_values.mean():,.0f}")
    print(f"  Median: ${np.median(home_values):,.0f}")
    print(f"  Range: ${home_values.min():,.0f} - ${home_values.max():,.0f}")
    
    return home_values

# ============================================================
# STEP 4: GENERATE HOME CHARACTERISTICS
# ============================================================

def generate_home_characteristics(home_values):
    """
    Generate square footage and lot size based on home value
    """
    print("\nGenerating home characteristics...")
    
    n_donors = len(home_values)
    
    base_sqft = 1000 + (home_values / 400) + np.random.normal(0, 300, n_donors)
    square_footage = np.clip(base_sqft, 500, 15000).astype(int)
    
    base_lot = square_footage * np.random.uniform(1.5, 8, n_donors)
    rural_boost = np.random.choice([0, 1], size=n_donors, p=[0.9, 0.1])
    base_lot = base_lot + rural_boost * np.random.uniform(20000, 100000, n_donors)
    lot_size = np.clip(base_lot, 1000, 500000).astype(int)
    
    print(f"  Square footage: {square_footage.mean():,.0f} avg")
    print(f"  Lot size: {lot_size.mean():,.0f} avg")
    
    return square_footage, lot_size

# ============================================================
# STEP 5: GENERATE LARGEST GIFT AMOUNTS
# ============================================================

def generate_largest_gifts(home_values, cap=1_000_000):
    """
    Generate largest gift amounts correlated with wealth
    """
    print("\nGenerating largest gift amounts...")
    
    n_donors = len(home_values)
    capacity = home_values * np.random.uniform(0.001, 0.05, n_donors)
    
    shape = 2.0
    scale = capacity / shape
    largest_gifts = np.random.gamma(shape, scale, n_donors)
    largest_gifts = np.clip(largest_gifts, 10, cap)
    
    print(f"  Mean: ${largest_gifts.mean():,.0f}")
    print(f"  Median: ${np.median(largest_gifts):,.0f}")
    print(f"  Correlation with home: {np.corrcoef(home_values, largest_gifts)[0,1]:.3f}")
    
    return largest_gifts

# ============================================================
# STEP 6: CALCULATE RELATIVE VALUES
# ============================================================

def calculate_relative_values(home_values, zip_medians, county_medians, state_medians):
    """Calculate ratios"""
    print("\nCalculating relative home values...")
    
    zip_ratios = home_values / zip_medians
    county_ratios = home_values / county_medians
    state_ratios = home_values / state_medians
    
    print(f"  ZIP ratios: {zip_ratios.mean():.2f} avg")
    print(f"  County ratios: {county_ratios.mean():.2f} avg")
    print(f"  State ratios: {state_ratios.mean():.2f} avg")
    
    return zip_ratios, county_ratios, state_ratios

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("DONOR WEALTH CAPACITY DATA GENERATOR")
    print("Using Census API for Real Geographic Data (FREE)")
    print("="*60)
    print(f"Generating {N_DONORS:,} donors\n")
    
    try:
        # Step 1: Fetch data
        state_medians = fetch_state_medians()
        county_medians = fetch_county_medians(max_counties=500)
        zip_medians = fetch_zip_medians(sample_size=1000)
        
        # Step 2: Assign geographies
        (donor_states, donor_zips, donor_counties,
         state_median_values, zip_median_values, 
         county_median_values) = assign_geographies(
            N_DONORS, state_medians, zip_medians, county_medians
        )
        
        # Step 3: Generate home values
        home_values = generate_home_values(N_DONORS, zip_median_values)
        
        # Step 4: Generate characteristics
        square_footage, lot_size = generate_home_characteristics(home_values)
        
        # Step 5: Generate gifts
        largest_gifts = generate_largest_gifts(home_values)
        
        # Step 6: Calculate ratios
        zip_ratios, county_ratios, state_ratios = calculate_relative_values(
            home_values, zip_median_values, county_median_values, state_median_values
        )
        
        # Step 7: Assemble dataset
        print("\nAssembling final dataset...")
        df = pd.DataFrame({
            'user_id': range(1, N_DONORS + 1),
            'largest_gift_amount': largest_gifts,
            'home_value_absolute': home_values,
            'home_value_zip_ratio': zip_ratios,
            'home_value_county_ratio': county_ratios,
            'home_value_state_ratio': state_ratios,
            'home_square_footage': square_footage,
            'lot_size_sqft': lot_size,
            'state_fips': donor_states,
            'zip_code': donor_zips,
            'county_name': donor_counties,
            'zip_median_home_value': zip_median_values,
            'county_median_home_value': county_median_values,
            'state_median_home_value': state_median_values
        })
        
        # Step 7.5: Convert to proper data types
        print("\nConverting to proper data types...")
        
        # Integer columns (whole numbers)
        int_cols = ['largest_gift_amount', 'home_value_absolute', 'home_square_footage', 
                    'lot_size_sqft', 'zip_median_home_value', 'county_median_home_value', 
                    'state_median_home_value']
        df[int_cols] = df[int_cols].round(0).astype(int)
        
        # Float columns (ratios with 4 decimal places)
        float_cols = ['home_value_zip_ratio', 'home_value_county_ratio', 'home_value_state_ratio']
        df[float_cols] = df[float_cols].round(4).astype(float)
        
        # Text columns (preserve leading zeros)
        df['state_fips'] = df['state_fips'].astype(str).str.zfill(2)
        df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
        
        # Step 8: Save
        # Step 8: Save with proper CSV quoting
        df.to_csv(OUTPUT_FILE, index=False, quoting=1)  # quoting=1 means QUOTE_ALL for non-numeric
        print(f"\n✓ Data saved to: {OUTPUT_FILE}")
                
        # Step 9: Summary
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(df[['largest_gift_amount', 'home_value_absolute', 
                  'home_value_zip_ratio', 'home_square_footage']].describe())
        
        print("\n" + "="*60)
        print("SAMPLE DATA")
        print("="*60)
        print(df.head(10))
        
        print("\n✓ COMPLETE!")
        
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Data generation failed")
        print("="*60)
        print(f"Error: {e}")
        print("\nPossible solutions:")
        print("1. Check internet connection")
        print("2. Get Census API key: https://api.census.gov/data/key_signup.html")
        print("3. Try again later")
        raise

if __name__ == "__main__":
    main()
