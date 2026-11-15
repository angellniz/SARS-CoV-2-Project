# Divergence-Dependent Novelty Detection for SARS-CoV-2 Genomic Surveillance

This repository contains the code and pipeline for the paper: 
**"Divergence-Dependent Novelty Detection for SARS-CoV-2 Genomic Surveillance Using k-mer Features and One-Class SVM"**

## Overview
This project evaluates a three-layer machine-learning pipeline for SARS-CoV-2 surveillance: 
1. Lineage classification (Random Forest, Linear SVM)
2. Similarity analysis (PCA, F-statistics)
3. Unsupervised novelty detection (One-Class SVM)

**Key Finding:** Novelty detection is a continuous function of phylogenetic divergence. Highly divergent lineages (Omicron, AUC = 0.93) are reliably flagged, while closely related lineages (Alpha, AUC = 0.47) fail.

## Data
The raw SARS-CoV-2 genomes were downloaded from NCBI GenBank using the `datasets` CLI. To reproduce the data pipeline, run the scripts in the following order:
1. `combine_dataset.py`
2. `filter_dataset.py`
3. `split_dataset.py`
4. `kmer_extract.py`

## Experiments
Run the following scripts to reproduce the results:
- **Experiment 1 (Classification):** `python train_experiment1.py`
- **Experiment 2 (k-mer Sweep):** `python train_experiment2.py`
- **Experiment 3 (Similarity/PCA):** `python experiment3_similarity.py`
- **Experiment 4 (Novelty Detection):** `python experiment4_novelty.py`

## Requirements
- Python 3.13
- `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`

## Reproducibility
All random seeds are fixed at 42. The pipeline runs end-to-end on a standard laptop CPU in ~20 minutes.