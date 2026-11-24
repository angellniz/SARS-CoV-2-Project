# Divergence-Dependent Novelty Detection for SARS-CoV-2 Genomic Surveillance

This repository contains the code, pipeline, and evaluation scripts for the paper: 
**"Divergence-Dependent Novelty Detection for SARS-CoV-2 Genomic Surveillance Using k-mer Features and One-Class SVM"**

##  Key Finding
Novelty detection in viral genomes is not a binary capability; it is a **continuous function of phylogenetic divergence**. Highly divergent lineages (Omicron BA.1, AUC = 0.93; Delta, AUC = 0.88) are reliably flagged as novel, while closely related lineages (Alpha, AUC = 0.47; Beta, AUC = 0.49) fail completely. 

##  Repository Structure
- `combine_dataset.py`: Merges downloaded FASTA files and NCBI metadata into a single CSV.
- `filter_dataset.py`: Applies quality control (length 29,000–30,000 bp, <5% ambiguous N bases).
- `split_dataset.py`: Generates random (80:20) and time-based train/test splits.
- `kmer_extract.py`: Extracts relative k-mer frequencies (k=3,4,5,6) and saves sparse matrices.
- `train_experiment1.py`: Trains RF and Linear SVM for baseline classification.
- `train_experiment2.py`: Runs the k-mer size sweep (k=3 to 6).
- `experiment3_similarity.py`: Computes lineage centroids, F-statistics, and PCA plots.
- `experiment4_novelty.py`: Trains One-Class SVM using a leave-one-lineage-out protocol.
- `experiment4_sensitivity.py`: Runs the novelty detection across different `nu` hyperparameters.

##  Requirements
The pipeline requires Python 3.13 (or 3.10+). Install the required packages via:
pip install numpy scipy pandas scikit-learn matplotlib

##  How to Reproduce
*Note: Because raw genomic data exceeds GitHub file limits, you must download the data from NCBI first.*

**Step 1: Download the Data**
Use the NCBI `datasets` CLI to download the 5 lineages (Alpha, Beta, Gamma, Delta, Omicron BA.1). Place the resulting FASTA files in this directory. (See the top of `combine_dataset.py` for the exact accession lists).

**Step 2: Process the Data**
Run the preprocessing scripts in order:
python combine_dataset.py
python filter_dataset.py
python split_dataset.py
python kmer_extract.py

**Step 3: Run the Experiments**
python train_experiment1.py     # Baseline Classification
python train_experiment2.py     # k-mer Sweep
python experiment3_similarity.py # PCA and F-statistics
python experiment4_novelty.py   # Novelty Detection (Main Contribution)
python experiment4_sensitivity.py # Nu parameter sweep

##  Results Summary
| Experiment | Task | Result |
| :--- | :--- | :--- |
| Exp 1 | Classification (RF & SVM) | **100% Accuracy** (Random & Time splits) |
| Exp 2 | k-mer Sweep | **k=4** is optimal (no gain at k=5 or 6) |
| Exp 3 | Similarity / PCA | Distinct clusters; signal distributed across all 256 features |
| Exp 4 | Novelty Detection (Omicron) | **AUC = 0.93** |
| Exp 4 | Novelty Detection (Alpha) | **AUC = 0.47** (Failed) |

##  Reproducibility
All random seeds are fixed at `42`. Every intermediate artifact (filtered datasets, split assignments, feature matrices, and predictions) is saved to disk in `.csv`, `.json`, `.npy`, and `.npz` formats. The pipeline runs end-to-end on a standard laptop CPU in approximately 20 minutes.


##  Contact
For questions or issues, please open an issue on GitHub or contact [angellunnss@gmail.com].
