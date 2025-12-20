"""
Donor Expected Value MCMC Model - APPLE SILICON (M1 MAX) GPU ACCELERATED
- Logistic regression for P(donation)
- Gamma regression for E[gift amount | donation]
- Expected value = P(donation) × E[gift amount]
- MODIFIED: Uses wealth capacity from INPUT_FILE_B (columns O and P)
- OPTIMIZED FOR: MacBook Pro M1 Max with Metal backend
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import time
from datetime import timedelta
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# ============================================================
# FILE PATHS - MAC OS
# ============================================================

INPUT_FILE_A = "/Users/dcurl/Desktop/Code/mcmc_artificial_data_main.csv"
INPUT_FILE_B = "/Users/dcurl/Desktop/Code/donor_wealth_capacity_data.csv"
OUTPUT_FILE = "/Users/dcurl/Desktop/Code/top_50_donors.csv"
TRACE_FILE = "/Users/dcurl/Desktop/Code/donor_trace.nc"
PLOT_FILE_COEFFICIENTS = "/Users/dcurl/Desktop/Code/coefficient_plot.png"
PLOT_FILE_TOP_DONORS = "/Users/dcurl/Desktop/Code/top_donors_plot.png"
PLOT_FILE_DIAGNOSTICS = "/Users/dcurl/Desktop/Code/mcmc_diagnostics.png"

# ============================================================
# GPU CONFIGURATION - APPLE SILICON (M1/M2/M3) SUPPORT
# ============================================================


def setup_gpu():
    """Configure JAX to use Apple Silicon GPU (Metal backend)."""

    try:
        import jax
        
        # For Apple Silicon, JAX uses Metal backend automatically
        # No need to set JAX_PLATFORM_NAME='gpu' on Mac
        devices = jax.devices()
        
        # Check if we have GPU devices
        gpu_available = any('gpu' in str(d).lower() or 'metal' in str(d).lower() for d in devices)
        
        if gpu_available:
            print(f"✓ APPLE SILICON GPU DETECTED: {devices}")
            print(f"  Using Metal acceleration on M1 Max\n")
            return True
        else:
            print(f"⚠ No GPU found. Available devices: {devices}")
            print(f"  Falling back to CPU\n")
            return False

    except ImportError:
        print("⚠ JAX not installed.")
        print("  Install with: pip install jax-metal")
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

def load_and_preprocess(filepath_a, filepath_b):
    """
    Load main donor data from FILE_A and merge with wealth capacity from FILE_B.
    Columns O and P from FILE_B contain capacity_score and capacity_std.
    """
    # Load main data (excluding capacity_score from column D)
    df_a = pd.read_csv(filepath_a)
    df_a.columns = df_a.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Load wealth capacity data
    df_b = pd.read_csv(filepath_b)
    # Assuming FILE_B has a donor ID in first column for merging
    # Columns O and P are indices 14 and 15 (0-indexed)
    capacity_cols = df_b.iloc[:, [0, 14, 15]]
    capacity_cols.columns = ['donor_id', 'capacity_score', 'capacity_std']
    
    # Merge the datasets
    # Assuming df_a has a donor_id column (adjust if different)
    if 'donor_id' not in df_a.columns:
        # If no donor_id, assume same row order
        df = df_a.copy()
        df['capacity_score'] = capacity_cols['capacity_score'].values
        df['capacity_std'] = capacity_cols['capacity_std'].values
    else:
        df = df_a.merge(capacity_cols, on='donor_id', how='left')
    
    # Process other columns as before
    df['gender_code'] = (df['gender'].str.lower() == 'm').astype(int)
    df['married'] = (df['marital_status'].str.lower() == 'y').astype(int)
    df['widowed'] = (df['marital_status'].str.lower() == 'w').astype(int)
    df['gave'] = (df['last_gift_amount'] > 0).astype(int)
    df['log_gift'] = np.where(df['last_gift_amount'] > 0,
                              np.log(df['last_gift_amount']),
                              np.nan)
    
    print(f"  Using capacity_score from INPUT_FILE_B")
    print(f"  Capacity score range: [{df['capacity_score'].min():.2f}, {df['capacity_score'].max():.2f}]")
    print(f"  Capacity std range: [{df['capacity_std'].min():.4f}, {df['capacity_std'].max():.4f}]")
    
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
    print(f"  Backend: JAX + NumPyro (Metal GPU-accelerated)\n")

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
    # Round all numeric outputs to 4 decimals
    results['ev_mean'] = np.round(ev_mean, 4)
    results['ev_std'] = np.round(ev_std, 4)
    results['ev_median'] = np.round(ev_median, 4)
    results['ev_q05'] = np.round(ev_q05, 4)
    results['ev_q95'] = np.round(ev_q95, 4)
    results['p_give_mean'] = np.round(p_give_samples.mean(axis=0), 4)
    results['mu_gift_mean'] = np.round(mu_gift_samples.mean(axis=0), 4)

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
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']].round(4))

    print("\n" + "="*60)
    print("GAMMA REGRESSION COEFFICIENTS (E[gift|give])")
    print("="*60)

    gamma_vars = ['alpha_gamma', 'gamma_recency', 'gamma_frequency', 'gamma_tenure',
                  'gamma_capacity', 'gamma_age', 'gamma_gender', 'gamma_married',
                  'gamma_widowed', 'shape_gamma']
    summary = az.summary(trace, var_names=gamma_vars)
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']].round(4))


# ============================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================

def create_coefficient_plot(trace, output_file):
    """Create a forest plot of coefficients with 95% HDI."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Logistic regression coefficients
    logit_vars = ['beta_recency', 'beta_frequency', 'beta_tenure',
                  'beta_capacity', 'beta_age', 'beta_gender', 'beta_married', 'beta_widowed']
    az.plot_forest(trace, var_names=logit_vars, combined=True, 
                   hdi_prob=0.95, ax=axes[0], colors='steelblue')
    axes[0].set_title('Logistic Regression Coefficients\n(Effect on P(Give))', 
                      fontsize=12, fontweight='bold')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Coefficient Value')
    
    # Gamma regression coefficients
    gamma_vars = ['gamma_recency', 'gamma_frequency', 'gamma_tenure',
                  'gamma_capacity', 'gamma_age', 'gamma_gender', 'gamma_married',
                  'gamma_widowed']
    az.plot_forest(trace, var_names=gamma_vars, combined=True, 
                   hdi_prob=0.95, ax=axes[1], colors='darkgreen')
    axes[1].set_title('Gamma Regression Coefficients\n(Effect on Gift Amount)', 
                      fontsize=12, fontweight='bold')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Coefficient Value')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Coefficient plot saved to: {output_file}")


def create_top_donors_plot(top_donors, output_file):
    """Create visualization of top 20 donors by expected value."""
    
    top_20 = top_donors.head(20)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Expected Value with uncertainty
    ax1 = axes[0, 0]
    x_pos = np.arange(len(top_20))
    ax1.barh(x_pos, top_20['ev_mean'], color='steelblue', alpha=0.7)
    ax1.errorbar(top_20['ev_mean'], x_pos, 
                 xerr=[top_20['ev_mean'] - top_20['ev_q05'], 
                       top_20['ev_q95'] - top_20['ev_mean']],
                 fmt='none', ecolor='black', capsize=3, alpha=0.5)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels([f"Rank {r}" for r in top_20['rank']])
    ax1.set_xlabel('Expected Value ($)')
    ax1.set_title('Top 20 Donors: Expected Value\n(with 90% Credible Interval)', 
                  fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: P(Give) vs Expected Gift Amount
    ax2 = axes[0, 1]
    scatter = ax2.scatter(top_20['p_give_mean'], top_20['mu_gift_mean'], 
                         c=top_20['ev_mean'], cmap='viridis', s=200, alpha=0.7)
    ax2.set_xlabel('Probability of Giving')
    ax2.set_ylabel('Expected Gift Amount ($)')
    ax2.set_title('P(Give) vs Expected Gift Amount\n(colored by Expected Value)', 
                  fontweight='bold')
    ax2.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Expected Value ($)')
    
    # Plot 3: Capacity Score vs Expected Value
    ax3 = axes[1, 0]
    ax3.scatter(top_20['capacity_score'], top_20['ev_mean'], 
               c='darkgreen', s=150, alpha=0.6)
    ax3.set_xlabel('Wealth Capacity Score')
    ax3.set_ylabel('Expected Value ($)')
    ax3.set_title('Capacity Score vs Expected Value', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Recency & Frequency
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(top_20['recency'], top_20['frequency'], 
                          c=top_20['ev_mean'], cmap='plasma', s=200, alpha=0.7)
    ax4.set_xlabel('Recency (days since last gift)')
    ax4.set_ylabel('Frequency (number of gifts)')
    ax4.set_title('Recency vs Frequency\n(colored by Expected Value)', 
                  fontweight='bold')
    ax4.grid(alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label('Expected Value ($)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Top donors plot saved to: {output_file}")


def create_diagnostics_plot(trace, output_file):
    """Create MCMC diagnostics plots (trace plots and R-hat)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Trace plots for key parameters
    key_vars = ['alpha_logit', 'alpha_gamma', 'beta_capacity', 'gamma_capacity']
    
    for idx, var in enumerate(key_vars):
        ax = axes[idx // 2, idx % 2]
        samples = trace.posterior[var].values
        for chain in range(samples.shape[0]):
            ax.plot(samples[chain, :], alpha=0.7, label=f'Chain {chain+1}')
        ax.set_title(f'Trace: {var}', fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        if idx == 0:
            ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostics plot saved to: {output_file}")


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # Check for GPU
    print("Checking for Apple Silicon GPU...")
    gpu_available = setup_gpu()

    timer = Timer()
    timer.start()

    # STEP 1: Load data
    print("Step 1/5: Loading data...")
    df = load_and_preprocess(INPUT_FILE_A, INPUT_FILE_B)
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
        print("  🚀 Using Metal GPU acceleration on M1 Max - this should be FAST!")

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
    print("\nStep 5/6: Generating donor rankings...")

    print("\n" + "="*60)
    print("TOP 50 DONORS BY EXPECTED VALUE PER CONTACT")
    print("="*60)

    top_donors = get_expected_values(trace, df, top_n=50)

    display_cols = ['rank', 'ev_mean', 'ev_q05', 'ev_q95', 'p_give_mean',
                    'mu_gift_mean', 'capacity_score', 'capacity_std', 'recency', 'frequency']

    print(top_donors[display_cols].to_string())

    # Save results
    top_donors.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    trace.to_netcdf(TRACE_FILE)
    print(f"MCMC trace saved to: {TRACE_FILE}")

    timer.mark("Results saved")

    # STEP 6: Generate visualizations
    print("\nStep 6/6: Generating visualizations...")
    
    create_coefficient_plot(trace, PLOT_FILE_COEFFICIENTS)
    create_top_donors_plot(top_donors, PLOT_FILE_TOP_DONORS)
    create_diagnostics_plot(trace, PLOT_FILE_DIAGNOSTICS)
    
    timer.mark("Visualizations created")

    # FINAL SUMMARY
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Total runtime: {timer.total_elapsed()}")
    print(f"Backend: {'Metal GPU (Apple Silicon)' if gpu_available else 'CPU'}")
    print(f"\nOutput files:")
    print(f"  Data:")
    print(f"    - {OUTPUT_FILE}")
    print(f"    - {TRACE_FILE}")
    print(f"  Visualizations:")
    print(f"    - {PLOT_FILE_COEFFICIENTS}")
    print(f"    - {PLOT_FILE_TOP_DONORS}")
    print(f"    - {PLOT_FILE_DIAGNOSTICS}")