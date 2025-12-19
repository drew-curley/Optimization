"""
Donor Expected Value MCMC Model - GPU ACCELERATED (CUDA) - LINUX/WSL2 VERSION
- Logistic regression for P(donation)
- Gamma regression for E[gift amount | donation]
- Expected value = P(donation) × E[gift amount]
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
from datetime import timedelta
import warnings
import os

# ============================================================
# FILE PATHS - WSL2 accesses Windows files via /mnt/c/
# ============================================================

INPUT_FILE = "/mnt/c/Users/dcurl/Desktop/Input/mcmc/donors.csv"
OUTPUT_FILE = "/mnt/c/Users/dcurl/Desktop/Input/mcmc/top_50_donors.csv"
TRACE_FILE = "/mnt/c/Users/dcurl/Desktop/Input/mcmc/donor_trace.nc"

# ============================================================
# GPU CONFIGURATION
# ============================================================


def setup_gpu():
    """Configure JAX to use GPU and verify CUDA is available."""

    os.environ['JAX_PLATFORM_NAME'] = 'gpu'

    try:
        import jax
        devices = jax.devices()
        gpu_available = any('cuda' in str(d).lower()
                            or 'gpu' in str(d).lower() for d in devices)

        if gpu_available:
            print(f"✓ GPU DETECTED: {devices}")
            print(f"  Using CUDA acceleration\n")
            return True
        else:
            print(f"⚠ No GPU found. Available devices: {devices}")
            print(f"  Falling back to CPU\n")
            return False

    except ImportError:
        print("⚠ JAX not installed.")
        return False


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================
# TIMING UTILITIES
# ============================================================

class Timer:
    def __init__(self):
        self.start_time = None
        self.step_times = {}

    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"MCMC MODEL STARTED AT {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

    def mark(self, step_name):
        elapsed = time.time() - self.start_time
        self.step_times[step_name] = elapsed
        print(f"[{self._format_time(elapsed)}] ✓ {step_name}")

    def total_elapsed(self):
        return self._format_time(time.time() - self.start_time)

    @staticmethod
    def _format_time(seconds):
        return str(timedelta(seconds=int(seconds)))


# ============================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    df['gender_code'] = (df['gender'].str.lower() == 'm').astype(int)
    df['married'] = (df['marital_status'].str.lower() == 'y').astype(int)
    df['widowed'] = (df['marital_status'].str.lower() == 'w').astype(int)
    df['gave'] = (df['last_gift_amount'] > 0).astype(int)
    df['log_gift'] = np.where(df['last_gift_amount'] > 0,
                              np.log(df['last_gift_amount']),
                              np.nan)
    return df


def standardize_predictors(df):
    continuous_vars = ['recency', 'frequency',
                       'tenure', 'capacity_score', 'age']
    means = {}
    stds = {}

    for var in continuous_vars:
        means[var] = df[var].mean()
        stds[var] = df[var].std()
        if stds[var] == 0:
            stds[var] = 1
        df[f'{var}_z'] = (df[var] - means[var]) / stds[var]

    return df, means, stds


# ============================================================
# 2. BUILD MCMC MODEL
# ============================================================

def build_model(df):
    n_donors = len(df)
    n_regions = df['region'].nunique()
    region_idx = df['region'].values - 1

    recency_z = df['recency_z'].values
    frequency_z = df['frequency_z'].values
    tenure_z = df['tenure_z'].values
    capacity_z = df['capacity_score_z'].values
    age_z = df['age_z'].values
    gender = df['gender_code'].values
    married = df['married'].values
    widowed = df['widowed'].values

    gave = df['gave'].values

    gave_mask = df['gave'] == 1
    gift_amounts = df.loc[gave_mask, 'last_gift_amount'].values
    gave_idx = np.where(gave_mask)[0]

    with pm.Model() as donor_model:

        # LOGISTIC REGRESSION: P(donation)
        alpha_logit = pm.Normal('alpha_logit', mu=0, sigma=2)
        beta_recency = pm.Normal('beta_recency', mu=0, sigma=1)
        beta_frequency = pm.Normal('beta_frequency', mu=0, sigma=1)
        beta_tenure = pm.Normal('beta_tenure', mu=0, sigma=1)
        beta_capacity = pm.Normal('beta_capacity', mu=0, sigma=1)
        beta_age = pm.Normal('beta_age', mu=0, sigma=1)
        beta_gender = pm.Normal('beta_gender', mu=0, sigma=1)
        beta_married = pm.Normal('beta_married', mu=0, sigma=1)
        beta_widowed = pm.Normal('beta_widowed', mu=0, sigma=1)

        sigma_region_logit = pm.HalfNormal('sigma_region_logit', sigma=1)
        region_effect_logit = pm.Normal('region_effect_logit', mu=0,
                                        sigma=sigma_region_logit, shape=n_regions)

        logit_p = (alpha_logit
                   + beta_recency * recency_z
                   + beta_frequency * frequency_z
                   + beta_tenure * tenure_z
                   + beta_capacity * capacity_z
                   + beta_age * age_z
                   + beta_gender * gender
                   + beta_married * married
                   + beta_widowed * widowed
                   + region_effect_logit[region_idx])

        p_give = pm.Deterministic('p_give', pm.math.sigmoid(logit_p))
        y_give = pm.Bernoulli('y_give', p=p_give, observed=gave)

        # GAMMA REGRESSION: E[gift amount | gave]
        alpha_gamma = pm.Normal('alpha_gamma', mu=8, sigma=2)
        gamma_recency = pm.Normal('gamma_recency', mu=0, sigma=0.5)
        gamma_frequency = pm.Normal('gamma_frequency', mu=0, sigma=0.5)
        gamma_tenure = pm.Normal('gamma_tenure', mu=0, sigma=0.5)
        gamma_capacity = pm.Normal('gamma_capacity', mu=0, sigma=0.5)
        gamma_age = pm.Normal('gamma_age', mu=0, sigma=0.5)
        gamma_gender = pm.Normal('gamma_gender', mu=0, sigma=0.5)
        gamma_married = pm.Normal('gamma_married', mu=0, sigma=0.5)
        gamma_widowed = pm.Normal('gamma_widowed', mu=0, sigma=0.5)

        sigma_region_gamma = pm.HalfNormal('sigma_region_gamma', sigma=0.5)
        region_effect_gamma = pm.Normal('region_effect_gamma', mu=0,
                                        sigma=sigma_region_gamma, shape=n_regions)

        log_mu_all = (alpha_gamma
                      + gamma_recency * recency_z
                      + gamma_frequency * frequency_z
                      + gamma_tenure * tenure_z
                      + gamma_capacity * capacity_z
                      + gamma_age * age_z
                      + gamma_gender * gender
                      + gamma_married * married
                      + gamma_widowed * widowed
                      + region_effect_gamma[region_idx])

        mu_gift_all = pm.Deterministic('mu_gift_all', pm.math.exp(log_mu_all))

        log_mu_gave = log_mu_all[gave_idx]
        mu_gift_gave = pm.math.exp(log_mu_gave)

        shape_gamma = pm.Exponential('shape_gamma', lam=0.1)
        rate_gave = shape_gamma / mu_gift_gave

        y_amount = pm.Gamma('y_amount', alpha=shape_gamma, beta=rate_gave,
                            observed=gift_amounts)

        expected_value = pm.Deterministic(
            'expected_value', p_give * mu_gift_all)

    return donor_model


# ============================================================
# 3. RUN INFERENCE (GPU-ACCELERATED)
# ============================================================

def run_inference_gpu(model, draws=2000, tune=1000, chains=4):
    print(f"\n  Configuration: {draws} draws, {tune} tuning, {chains} chains")
    print(f"  Total samples: {draws * chains:,}")
    print(f"  Backend: JAX + NumPyro (GPU-accelerated)\n")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            return_inferencedata=True,
            random_seed=42,
            progressbar=True,
            nuts_sampler='numpyro',
        )
    return trace


def run_inference_cpu(model, draws=2000, tune=1000, chains=4):
    print(f"\n  Configuration: {draws} draws, {tune} tuning, {chains} chains")
    print(f"  Total samples: {draws * chains:,}")
    print(f"  Backend: PyTensor (CPU)\n")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            return_inferencedata=True,
            random_seed=42,
            progressbar=True
        )
    return trace


# ============================================================
# 4. EXTRACT RESULTS AND RANK DONORS
# ============================================================

def get_expected_values(trace, df, top_n=50):
    ev_samples = trace.posterior['expected_value'].values
    n_chains, n_draws, n_donors = ev_samples.shape
    ev_flat = ev_samples.reshape(n_chains * n_draws, n_donors)

    ev_mean = ev_flat.mean(axis=0)
    ev_std = ev_flat.std(axis=0)
    ev_median = np.median(ev_flat, axis=0)
    ev_q05 = np.percentile(ev_flat, 5, axis=0)
    ev_q95 = np.percentile(ev_flat, 95, axis=0)

    p_give_samples = trace.posterior['p_give'].values.reshape(
        n_chains * n_draws, n_donors)
    mu_gift_samples = trace.posterior['mu_gift_all'].values.reshape(
        n_chains * n_draws, n_donors)

    results = df.copy()
    results['ev_mean'] = ev_mean
    results['ev_std'] = ev_std
    results['ev_median'] = ev_median
    results['ev_q05'] = ev_q05
    results['ev_q95'] = ev_q95
    results['p_give_mean'] = p_give_samples.mean(axis=0)
    results['mu_gift_mean'] = mu_gift_samples.mean(axis=0)

    results['rank'] = results['ev_mean'].rank(ascending=False).astype(int)
    results = results.sort_values('rank')

    return results.head(top_n)


def print_coefficient_summary(trace):
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION COEFFICIENTS (P(give))")
    print("="*60)

    logit_vars = ['alpha_logit', 'beta_recency', 'beta_frequency', 'beta_tenure',
                  'beta_capacity', 'beta_age', 'beta_gender', 'beta_married', 'beta_widowed']
    summary = az.summary(trace, var_names=logit_vars)
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])

    print("\n" + "="*60)
    print("GAMMA REGRESSION COEFFICIENTS (E[gift|give])")
    print("="*60)

    gamma_vars = ['alpha_gamma', 'gamma_recency', 'gamma_frequency', 'gamma_tenure',
                  'gamma_capacity', 'gamma_age', 'gamma_gender', 'gamma_married',
                  'gamma_widowed', 'shape_gamma']
    summary = az.summary(trace, var_names=gamma_vars)
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # Check for GPU
    print("Checking for GPU...")
    gpu_available = setup_gpu()

    timer = Timer()
    timer.start()

    # STEP 1: Load data
    print("Step 1/5: Loading data...")
    df = load_and_preprocess(INPUT_FILE)
    df, means, stds = standardize_predictors(df)

    print(f"  Loaded {len(df):,} donors")
    print(
        f"  Donors who gave: {df['gave'].sum():,} ({100*df['gave'].mean():.1f}%)")
    if df['gave'].sum() > 0:
        print(
            f"  Average gift (among givers): ${df[df['gave'] == 1]['last_gift_amount'].mean():,.2f}")
    timer.mark("Data loaded and preprocessed")

    # STEP 2: Build model
    print("\nStep 2/5: Building model...")
    model = build_model(df)
    timer.mark("Model built")

    # STEP 3: Run MCMC
    print("\nStep 3/5: Running MCMC sampling...")
    if gpu_available:
        print("  🚀 Using GPU acceleration - this should be FAST!")

    mcmc_start = time.time()

    if gpu_available:
        trace = run_inference_gpu(model, draws=2000, tune=1000, chains=4)
    else:
        try:
            trace = run_inference_gpu(model, draws=2000, tune=1000, chains=4)
        except Exception as e:
            print(f"  NumPyro failed ({e}), using PyTensor...")
            trace = run_inference_cpu(model, draws=2000, tune=1000, chains=4)

    mcmc_elapsed = time.time() - mcmc_start
    timer.mark(f"MCMC sampling complete ({int(mcmc_elapsed)}s)")

    # STEP 4: Analyze results
    print("\nStep 4/5: Analyzing results...")

    print("\n" + "="*60)
    print("MCMC DIAGNOSTICS")
    print("="*60)
    print(az.summary(trace, var_names=['alpha_logit', 'alpha_gamma']))

    print_coefficient_summary(trace)
    timer.mark("Results analyzed")

    # STEP 5: Generate rankings and save
    print("\nStep 5/5: Generating donor rankings...")

    print("\n" + "="*60)
    print("TOP 50 DONORS BY EXPECTED VALUE PER CONTACT")
    print("="*60)

    top_donors = get_expected_values(trace, df, top_n=50)

    display_cols = ['rank', 'ev_mean', 'ev_q05', 'ev_q95', 'p_give_mean',
                    'mu_gift_mean', 'capacity_score', 'recency', 'frequency']

    print(top_donors[display_cols].to_string())

    # Save results
    top_donors.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    trace.to_netcdf(TRACE_FILE)
    print(f"MCMC trace saved to: {TRACE_FILE}")

    timer.mark("Results saved")

    # FINAL SUMMARY
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Total runtime: {timer.total_elapsed()}")
    print(f"Backend: {'GPU (CUDA)' if gpu_available else 'CPU'}")
    print(f"\nOutput files (accessible from Windows):")
    print(f"  - C:\\Users\\dcurl\\Desktop\\Input\\mcmc\\top_50_donors.csv")
    print(f"  - C:\\Users\\dcurl\\Desktop\\Input\\mcmc\\donor_trace.nc")
