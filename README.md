# Public reproducibility package: PINN pseudo-velocity in pancreatic cancer

This repository is the minimal, de-identified reproducibility package for a physics-informed neural network (PINN) fitted to total-mRNA single-cell RNA-sequencing data from pancreatic cancer atlases. The model estimates an ODE-constrained **pseudo-velocity** along diffusion pseudotime; it is not conventional splicing-based RNA velocity and does not measure chronological dynamics.

## Scope

The package contains the processed 986-cell, 53-gene training matrix; a fitted checkpoint and loss history; de-identified TCGA-PAAD survival inputs for 182 cases; figure-level source tables; and scripts that retrain the PINN and reproduce the univariable survival analyses. Raw sequencing matrices, source-study accession files, manuscript drafts, credentials, and private development assets are deliberately excluded.

The governing system is:

```text
dx/dt = alpha_theta(x) - gamma * x
```

The physics residual uses exact per-gene derivatives. Diffusion pseudotime supplies an inferred ordering, not observed time.

## Repository layout

```text
data/                 de-identified analysis and figure-source CSV files
model/                reported fitted checkpoint and training-loss history
scripts/              PINN retraining, TCGA validation, and integrity checks
outputs/              generated results (ignored except for its placeholder)
checksums.sha256       SHA-256 manifest for published data and model assets
requirements.txt      compact Python environment specification
```

`cell_id` and `case_id` are deterministic, study-local row labels created for this public package. They are not source accession identifiers and cannot be used to recover the removed TCGA barcodes.

## Reproduce the reported validation

```bash
git clone https://github.com/Javad-Alizargar/PINN_scRNA_Pancreatic_Velocity_Public.git
cd PINN_scRNA_Pancreatic_Velocity_Public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_package.py
```

The verification command checks schemas, identifier removal, file checksums, the continuous Cox result, the upper-quartile hazard ratio, and the 201-day median-survival decrement. It writes regenerated survival statistics to `outputs/tcga_survival_statistics.json`.

## Retrain the PINN

The reported optimization used a fixed seed, 5,000 epochs, a learning rate of `1e-3`, equal data and physics-loss weights, and two 128-unit hidden layers in each network.

```bash
python scripts/train_pinn.py --epochs 5000 --device auto
```

Training outputs are written to `outputs/pinn_sc_weights.pt` and `outputs/pinn_loss_history.csv`. CPU execution is supported but slower. Hardware-dependent floating-point differences may prevent bitwise identity while preserving the specified model and loss formulation.

## Reproduce TCGA-PAAD statistics only

```bash
python scripts/validate_tcga.py
```

The five-gene squamous-associated score comprises `KRT17`, `EMP1`, `S100A2`, `KRT16`, and `S100A4`. All survival models are univariable and support association—not clinical independence, prospective validity, or causation.

## Data provenance and interpretation

The single-cell inputs derive from GEO series GSE155698 and GSE165399. The survival analysis derives from TCGA-PAAD data distributed through UCSC Xena. Public tables contain processed values only. Consult the manuscript and source studies for cohort selection, preprocessing, biological interpretation, and primary-data terms of use.

No IPMN cells passed the epithelial gate used for this fitted matrix, and PASC cells contribute substantially to the squamous-associated signal. These limitations should accompany reuse of the model outputs.

## Citation and archive

The Zenodo DOI and final article citation will be inserted after archival release. Until then, cite this GitHub repository and retain the commit identifier used for analysis.

