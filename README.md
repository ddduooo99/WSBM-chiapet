# WSBM-chiapet

This repository contains the implementation of a probabilistic loop detection model for ChIA-PET data, based on the **Weighted Stochastic Block Model (WSBM)** framework. The model performs data-driven community detection to identify pairwise and multi-way chromatin interactions without relying on anchor definitions or peak calling.

## Features

- Graph-based modeling of PETs (Paired-End Tags)
- Probabilistic inference of chromatin loops
- Integration of prior biological constraints (e.g., read depth, fragment length)
- Code modularized for sampling and inference
