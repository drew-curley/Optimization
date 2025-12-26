## Repository Contents

### `donors.csv`

A **synthetic dataset containing 10,000 artificial donor records** (default).

Each row represents a single donor, and each column corresponds to a feature used in the modeling process, including:

* Recency
* Frequency
* Tenure
* Capacity score
* Gift history
* Age
* Region
* Additional demographic and behavioral variables

This file serves as a **standalone donor table** suitable for testing donor models, optimization routines, and MCMC workflows without using real donor data.

---

### `mcmc_artificial_data_generators.py`

A Python script for **generating artificial donor data**.

This script is intended for:

* Simulation workflows
* Rapid experimentation
* Model testing without reliance on real donor information

It assigns a **Wealth Capacity Score (WCS)** directly as a latent indicator of donor potential.
It **does not** generate home values or geographic features.

**Output:**
`donors_artificial_data_main.csv`

---

### Wealth Capacity Score (WCS)

This repository uses a **Wealth Capacity Score (WCS)** as a unified, interpretable metric representing a donor’s potential wealth and capacity for large gifts.

The WCS aggregates key behavioral and financial indicators into a single score designed to act as a **latent proxy for donor wealth**. While synthetic, it mirrors how capacity scores are used in real-world fundraising analytics.

---

### `data_generator_for_wealth_capacity_score.py`

A Python script for generating **artificial donor wealth data** used to construct the Wealth Capacity Score.

This script:

* Assigns **home values**
* Assigns **geographic locations**
* Derives a **Wealth Capacity Score** from those inputs

It is designed for testing and simulation workflows that require **externally grounded wealth signals** (e.g., housing-based proxies).

**Output:**
`mcmc_artificial_data_main.csv`

---

## MCMC Model Versions

### `mcmcv1`

The **initial conceptual implementation**.

A minimal, exploratory version demonstrating the core idea of applying MCMC methods to donor modeling and value estimation.

---

### `mcmcv2`

The **first fully functioning MCMC model**.

Includes:

* A complete sampling workflow
* Explicit likelihood definitions
* Integration of donor features into the probabilistic model

---

### `mcmcv3`

A **rewritten MCMC model optimized for CUDA-enabled GPUs**.

Designed to achieve significant performance improvements when working with:

* Large donor datasets
* Higher-dimensional parameter spaces
* More computationally intensive sampling strategies

---

### `mcmcv4`

An adaptation of the MCMC model for **Apple Silicon** environments.

This version is functionally incomplete due to incorrect or hard-coded file paths and is retained primarily for reference.

---

### `mcmcv5`

The **current, fully functional main version**.

Key characteristics:

* Bifurcated architecture
* MCMC sampling executed in a separate embedded script
* Clean separation between data generation, inference, and ranking

This version serves as the **primary implementation used by the repository**.

---

### `mcmcv6` (Planned)

A future version intended to:

* Recombine bifurcated components
* Consolidate MCMC execution into a single, unified pipeline
* Improve maintainability and extensibility

