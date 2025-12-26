Repository Contents
donors.csv

A synthetic dataset containing 10,000 artificial donor records (default).
Each row represents one donor, and each column corresponds to a feature used in modeling—such as recency, frequency, tenure, capacity score, gift history, age, region, and more.

mcmc_artificial_data_generators.py

A Python script for generating artificial donor data.
Useful for testing, simulation workflows, and rapid experiments without relying on real donor information. It assigns Wealth Capcity Score, does not generate. OUtput saved as donors_artificial_data_main.csv

Wealth Capacity Score

This repository uses a Wealth Capacity Score (WCS) as a unified metric representing a donor’s potential wealth and capacity for large gifts.
The formula combines key behavioral and financial indicators into a single interpretable score.

data_generator_fpr_wealth_capacity_score.py

A Python script for generating artificial donor Wealth Capacity scrore data.
Useful for testing, simulation workflows, and rapid experiments without relying on real donor information. It assigns home values and locations, then generates Wealth Capcity Score. Output saved as mcmc_artificial_data_main.csv.

MCMC Model Versions
mcmcv1

The initial concept implementation.
A minimal, exploratory version demonstrating the basic ideas behind using MCMC for donor modeling.

mcmcv2

The first fully functioning MCMC model.
Includes a complete sampling workflow, likelihood definitions, and feature integration.

mcmcv3

A rewritten MCMC model designed to run on CUDA-enabled GPUs for significant performance improvements in large-scale datasets or more complex parameter spaces.

mcmcv4 

Adapted to silicon, but bad file paths

mcmcv5
Fully-functional, bifrucated. Embedded MCMC runs first on separate script. mcmcv6 to combine...
