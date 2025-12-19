"""
Donor Expected Value MCMC Model
- Logistic regression for P(donation)
- Gamma regression for E[gift amount | donation]
- Expected value = P(donation) × E[gift amount]
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats
import time
from datetime import timedelta
import warnings

# Suppress PyTensor C compiler warnings (model still works, just slower)
warnings.filterwarnings('ignore', category=UserWarning, module='pytensor')

# ============================================================
# TIMING UTILITIES
# ============================================================


class Timer:
    """Simple timer class for tracking elapsed and estimated remaining time."""

    def __init__(self):
        self.start_time = None
        self.step_times = {}

    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"MCMC MODEL STARTED AT {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

    def mark(self, step_name):
        """Mark completion of a step."""
        elapsed = time.time() - self.start_time
        self.step_times[step_name] = elapsed
        print(f"[{self._format_time(elapsed)}] ✓ {step_name}")

    def estimate_remaining(self, current_step, total_steps):
        """Estimate remaining time based on progress."""
        elapsed = time.time() - self.start_time
        if current_step > 0:
            estimated_total = elapsed * total_steps / current_step
            remaining = estimated_total - elapsed
            return self._format_time(remaining)
        return "calculating..."

    def total_elapsed(self):
        """Get total elapsed time."""
        return self._format_time(time.time() - self.start_time)

    @staticmethod
    def _format_time(seconds):
        """Format seconds as HH:MM:SS."""
        return str(timedelta(seconds=int(seconds)))


# ============================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================

def load_and_preprocess(filepath):
    """Load CSV and encode categorical variables."""
    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # Encode categorical variables
    df['gender_code'] = (df['gender'].str.lower() ==
                         'm').astype(int)  # 1=male, 0=female

    # Marital status: one-hot encode (y=married, n=not married, w=widowed)
    df['married'] = (df['marital_status'].str.lower() == 'y').astype(int)
    df['widowed'] = (df['marital_status'].str.lower() == 'w').astype(int)

    # Create binary outcome: did donor give? (last_gift_amount > 0)
    df['gave'] = (df['last_gift_amount'] > 0).astype(int)

    # Log transform gift amount for those who gave (for gamma modeling)
    df['log_gift'] = np.where(df['last_gift_amount'] > 0,
                              np.log(df['last_gift_amount']),
                              np.nan)

    return df


def standardize_predictors(df):
    """Standardize continuous predictors for better MCMC sampling."""
    continuous_vars = ['recency', 'frequency',
                       'tenure', 'capacity_score', 'age']

    means = {}
    stds = {}

    for var in continuous_vars:
        means[var] = df[var].mean()
        stds[var] = df[var].std()
        # Handle zero std (constant column)
        if stds[var] == 0:
            stds[var] = 1
        df[f'{var}_z'] = (df[var] - means[var]) / stds[var]

    return df, means, stds


# ============================================================
# 2. BUILD MCMC MODEL
# ============================================================

def build_model(df):
    """
    Build PyMC model with:
    - Logistic regression for P(give)
    - Gamma regression for gift amount conditional on giving
    """

    # Prepare data
    n_donors = len(df)
    n_regions = df['region'].nunique()

    # Indices for region random effects
    region_idx = df['region'].values - 1  # 0-indexed

    # Predictors (standardized)
    recency_z = df['recency_z'].values
    frequency_z = df['frequency_z'].values
    tenure_z = df['tenure_z'].values
    capacity_z = df['capacity_score_z'].values
    age_z = df['age_z'].values
    gender = df['gender_code'].values
    married = df['married'].values
    widowed = df['widowed'].values

    # Outcomes
    gave = df['gave'].values

    # For gamma model: only donors who gave
    gave_mask = df['gave'] == 1
    gift_amounts = df.loc[gave_mask, 'last_gift_amount'].values
    gave_idx = np.where(gave_mask)[0]

    with pm.Model() as donor_model:

        # --------------------------------------------------------
        # LOGISTIC REGRESSION: P(donation)
        # --------------------------------------------------------

        # Priors for logistic regression coefficients
        alpha_logit = pm.Normal('alpha_logit', mu=0, sigma=2)

        beta_recency = pm.Normal('beta_recency', mu=0, sigma=1)
        beta_frequency = pm.Normal('beta_frequency', mu=0, sigma=1)
        beta_tenure = pm.Normal('beta_tenure', mu=0, sigma=1)
        beta_capacity = pm.Normal('beta_capacity', mu=0, sigma=1)
        beta_age = pm.Normal('beta_age', mu=0, sigma=1)
        beta_gender = pm.Normal('beta_gender', mu=0, sigma=1)
        beta_married = pm.Normal('beta_married', mu=0, sigma=1)
        beta_widowed = pm.Normal('beta_widowed', mu=0, sigma=1)

        # Region random effects
        sigma_region_logit = pm.HalfNormal('sigma_region_logit', sigma=1)
        region_effect_logit = pm.Normal('region_effect_logit', mu=0,
                                        sigma=sigma_region_logit, shape=n_regions)

        # Linear predictor for logistic
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

        # Probability of giving
        p_give = pm.Deterministic('p_give', pm.math.sigmoid(logit_p))

        # Likelihood for giving (binary)
        y_give = pm.Bernoulli('y_give', p=p_give, observed=gave)

        # --------------------------------------------------------
        # GAMMA REGRESSION: E[gift amount | gave]
        # --------------------------------------------------------

        # Priors for gamma regression coefficients
        alpha_gamma = pm.Normal('alpha_gamma', mu=8,
                                sigma=2)  # log scale (~$3000)

        gamma_recency = pm.Normal('gamma_recency', mu=0, sigma=0.5)
        gamma_frequency = pm.Normal('gamma_frequency', mu=0, sigma=0.5)
        gamma_tenure = pm.Normal('gamma_tenure', mu=0, sigma=0.5)
        gamma_capacity = pm.Normal('gamma_capacity', mu=0, sigma=0.5)
        gamma_age = pm.Normal('gamma_age', mu=0, sigma=0.5)
        gamma_gender = pm.Normal('gamma_gender', mu=0, sigma=0.5)
        gamma_married = pm.Normal('gamma_married', mu=0, sigma=0.5)
        gamma_widowed = pm.Normal('gamma_widowed', mu=0, sigma=0.5)

        # Region random effects for gamma
        sigma_region_gamma = pm.HalfNormal('sigma_region_gamma', sigma=0.5)
        region_effect_gamma = pm.Normal('region_effect_gamma', mu=0,
                                        sigma=sigma_region_gamma, shape=n_regions)

        # Linear predictor for log(mu) of gamma (for all donors)
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

        # Expected gift amount for all donors (used for expected value calc)
        mu_gift_all = pm.Deterministic('mu_gift_all', pm.math.exp(log_mu_all))

        # For likelihood: only use donors who gave
        log_mu_gave = log_mu_all[gave_idx]
        mu_gift_gave = pm.math.exp(log_mu_gave)

        # Shape parameter for gamma (controls variance)
        shape_gamma = pm.Exponential('shape_gamma', lam=0.1)

        # Rate = shape / mu (parameterization: mean = shape/rate)
        rate_gave = shape_gamma / mu_gift_gave

        # Likelihood for gift amount (only for those who gave)
        y_amount = pm.Gamma('y_amount', alpha=shape_gamma, beta=rate_gave,
                            observed=gift_amounts)

        # --------------------------------------------------------
        # EXPECTED VALUE PER CONTACT
        # --------------------------------------------------------

        expected_value = pm.Deterministic(
            'expected_value', p_give * mu_gift_all)

    return donor_model


# ============================================================
# 3. RUN INFERENCE
# ============================================================

def run_inference(model, draws=2000, tune=1000, chains=4):
    """Run MCMC sampling with progress callback."""
    print(f"\n  Configuration: {draws} draws, {tune} tuning, {chains} chains")
    print(f"  Total samples: {draws * chains:,}")
    print(f"  Note: Without C compiler, sampling will be slower but still accurate.\n")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            return_inferencedata=True,
            random_seed=42,
            progressbar=True  # Shows PyMC's built-in progress bar
        )
    return trace


# ============================================================
# 4. EXTRACT RESULTS AND RANK DONORS
# ============================================================

def get_expected_values(trace, df, top_n=50):
    """Extract posterior expected values and rank donors."""

    # Get posterior samples of expected value
    ev_samples = trace.posterior['expected_value'].values

    # Reshape: (chains, draws, donors) -> (samples, donors)
    n_chains, n_draws, n_donors = ev_samples.shape
    ev_flat = ev_samples.reshape(n_chains * n_draws, n_donors)

    # Summary statistics
    ev_mean = ev_flat.mean(axis=0)
    ev_std = ev_flat.std(axis=0)
    ev_median = np.median(ev_flat, axis=0)
    ev_q05 = np.percentile(ev_flat, 5, axis=0)
    ev_q95 = np.percentile(ev_flat, 95, axis=0)

    # Also get P(give) and E[gift|give] separately
    p_give_samples = trace.posterior['p_give'].values.reshape(
        n_chains * n_draws, n_donors)
    mu_gift_samples = trace.posterior['mu_gift_all'].values.reshape(
        n_chains * n_draws, n_donors)

    # Create results dataframe
    results = df.copy()
    results['ev_mean'] = ev_mean
    results['ev_std'] = ev_std
    results['ev_median'] = ev_median
    results['ev_q05'] = ev_q05
    results['ev_q95'] = ev_q95
    results['p_give_mean'] = p_give_samples.mean(axis=0)
    results['mu_gift_mean'] = mu_gift_samples.mean(axis=0)

    # Rank by expected value
    results['rank'] = results['ev_mean'].rank(ascending=False).astype(int)
    results = results.sort_values('rank')

    return results.head(top_n)


def print_coefficient_summary(trace):
    """Print summary of model coefficients."""

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

    # Initialize timer
    timer = Timer()
    timer.start()

    # --------------------------------------------------------
    # STEP 1: Load data
    # --------------------------------------------------------
    print("Step 1/5: Loading data...")
    df = load_and_preprocess(r'C:\Users\dcurl\Desktop\Input\mcmc\donors.csv')
    df, means, stds = standardize_predictors(df)

    print(f"  Loaded {len(df):,} donors")
    print(
        f"  Donors who gave: {df['gave'].sum():,} ({100*df['gave'].mean():.1f}%)")
    print(
        f"  Average gift (among givers): ${df[df['gave'] == 1]['last_gift_amount'].mean():,.2f}")
    timer.mark("Data loaded and preprocessed")

    # --------------------------------------------------------
    # STEP 2: Build model
    # --------------------------------------------------------
    print("\nStep 2/5: Building model...")
    model = build_model(df)
    timer.mark("Model built")

    # --------------------------------------------------------
    # STEP 3: Run MCMC (this is the slow part)
    # --------------------------------------------------------
    print("\nStep 3/5: Running MCMC sampling...")
    print("  (This is the longest step - typically 5-30 minutes depending on data size)")

    mcmc_start = time.time()
    trace = run_inference(model, draws=2000, tune=1000, chains=4)
    mcmc_elapsed = time.time() - mcmc_start

    timer.mark(f"MCMC sampling complete ({int(mcmc_elapsed)}s)")

    # --------------------------------------------------------
    # STEP 4: Analyze results
    # --------------------------------------------------------
    print("\nStep 4/5: Analyzing results...")

    # Print diagnostics
    print("\n" + "="*60)
    print("MCMC DIAGNOSTICS")
    print("="*60)
    print(az.summary(trace, var_names=['alpha_logit', 'alpha_gamma']))

    # Print coefficient summary
    print_coefficient_summary(trace)

    timer.mark("Results analyzed")

    # --------------------------------------------------------
    # STEP 5: Generate rankings and save
    # --------------------------------------------------------
    print("\nStep 5/5: Generating donor rankings...")

    # Get ranked expected values
    print("\n" + "="*60)
    print("TOP 50 DONORS BY EXPECTED VALUE PER CONTACT")
    print("="*60)

    top_donors = get_expected_values(trace, df, top_n=50)

    # Display key columns
    display_cols = ['rank', 'ev_mean', 'ev_q05', 'ev_q95', 'p_give_mean',
                    'mu_gift_mean', 'capacity_score', 'recency', 'frequency']

    print(top_donors[display_cols].to_string())

    # Save full results
    output_path = r'C:\Users\dcurl\Desktop\Input\mcmc\top_50_donors.csv'
    top_donors.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Save trace for later analysis
    trace_path = r'C:\Users\dcurl\Desktop\Input\mcmc\donor_trace.nc'
    trace.to_netcdf(trace_path)
    print(f"MCMC trace saved to: {trace_path}")

    timer.mark("Results saved")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Total runtime: {timer.total_elapsed()}")
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {trace_path}")
