# MCMC Donor Capacity Modeling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements Markov Chain Monte Carlo (MCMC) models to estimate donors' **Wealth Capacity Scores**—a probabilistic quantification of a donor's potential wealth and capacity for large gifts. Built for fundraising analytics, the models use artificial donor data to simulate real-world scenarios, enabling rapid testing and iteration without privacy concerns.

Key highlights:
- **Scalable MCMC Sampling**: From basic prototypes to GPU-accelerated inference.
- **Artificial Data Pipeline**: Generate customizable datasets mimicking donor demographics, giving history, and engagement metrics.
- **Wealth Capacity Score Formula**: A Bayesian-derived score blending prior wealth indicators with posterior gift propensity.

Ideal for nonprofits, data scientists, or researchers exploring donor behavior modeling.

## Features

- **Data Generation**: Script to create synthetic donor datasets (default: 10,000 records).
- **Model Versions**:
  - **MCMCv1**: Conceptual prototype demonstrating core MCMC logic.
  - **MCMCv2**: Production-ready model with full sampling and score computation.
  - **MCMCv3**: CUDA-optimized for faster inference on GPUs.
- **Modular Design**: Easy to extend features (e.g., add real data integration).
- **Evaluation Tools**: Built-in metrics for model convergence and score validation.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mcmc-donor-modeling.git
   cd mcmc-donor-modeling
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   Core dependencies include:
   - `numpy`, `pandas` for data handling.
   - `pymc` or `torch` for MCMC (version-specific).
   - `matplotlib` for visualizations.

   For MCMCv3 (CUDA support):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Quick Start

### 1. Generate Artificial Data
Use the data generator to create `donors.csv` (default: 10,000 donors).

```bash
python mcmc_artificial_data_generators.py --num_donors 10000 --output donors.csv
```

**Dataset Schema** (`donors.csv` columns as features):
| Column              | Description                          | Type    | Example Range |
|---------------------|--------------------------------------|---------|---------------|
| `age`              | Donor's age                          | int    | 18–90        |
| `income`           | Annual income (imputed)              | float  | 0–500k       |
| `past_gifts_total` | Sum of historical donations          | float  | 0–100k       |
| `engagement_score` | Interaction frequency (e.g., emails) | float  | 0–10         |
| `zip_code`         | Location proxy for wealth proxies    | str    | US ZIP       |
| ... (extensible)   | Additional features like `net_worth_estimate` | varies | Varies       |

This synthetic data simulates realistic correlations (e.g., higher income → larger gifts).

### 2. Run Models
#### MCMCv1: Basic Idea (Proof-of-Concept)
Explores core MCMC setup without optimizations. Run for quick ideation:
```bash
python mcmcv1.py --data donors.csv --samples 1000
```
- Outputs: Trace plots and basic score distributions.
- Use case: Validate assumptions on small datasets.

#### MCMCv2: First Functioning Model
Full Bayesian inference with priors on wealth capacity. Produces reliable scores:
```bash
python mcmcv2.py --data donors.csv --samples 5000 --chains 4
```
- Outputs: `scores.csv` with per-donor Wealth Capacity Scores (0–1 scale, where >0.8 indicates high potential).
- **Wealth Capacity Score Formula**:
  \[
  WCS = \int P(\text{Gift} > \theta \mid \mathbf{X}) \cdot \pi(\mathbf{X}) \, d\mathbf{X}
  \]
  Where:
  - \(\mathbf{X}\): Donor features (e.g., income, past gifts).
  - \(\pi(\mathbf{X})\): Prior distribution (e.g., log-normal for wealth).
  - \(\theta\): Threshold for "large gift" (configurable, default $10k).
  - Computed via MCMC posterior sampling for uncertainty quantification.

#### MCMCv3: CUDA-Accelerated Model
GPU version of v2 for large-scale runs (10k+ donors):
```bash
python mcmcv3.py --data donors.csv --samples 10000 --device cuda
```
- Requires NVIDIA GPU with CUDA 11.8+.
- Speedup: ~5–10x faster than CPU for high-sample counts.
- Outputs: Same as v2, plus GPU utilization logs.

### 3. Visualize Results
Generate plots for score distributions:
```bash
python visualize_scores.py --input scores.csv
```
- Produces: Histograms, donor segmentation charts, and convergence diagnostics.

## Model Details

- **MCMCv1**: Focuses on univariate priors (e.g., income → capacity). Non-parallel, for learning.
- **MCMCv2**: Multivariate hierarchical model incorporating all features. Uses NUTS sampler for efficiency.
- **MCMCv3**: Ports to PyTorch for GPU tensor operations. Maintains v2 logic but accelerates sampling loops.

All models output donor rankings by WCS, helping prioritize outreach for major gifts.

## Testing & Customization
- **Unit Tests**: Run `pytest` to validate data gen and model outputs.
- **Extend Data**: Modify `mcmc_artificial_data_generators.py` to add features (e.g., integrate real anonymized data).
- **Hyperparameters**: Tune via CLI flags (e.g., `--burnin 1000` for MCMC warmup).

## Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit changes (`git commit -m 'Add some amazing feature'`).
4. Push to branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Feedback on model accuracy or new features welcome!

## License
MIT License—feel free to use, modify, and distribute.

## Acknowledgments
Inspired by Bayesian fundraising analytics. Built with PyMC/PyTorch for robust inference.

---

*Last updated: December 1, 2025*  
Questions? Open an issue! 🚀
