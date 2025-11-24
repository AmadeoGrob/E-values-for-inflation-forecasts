# Anytime Valid Inference for Probabilistic Forecasts

This repository contains the complete code, data, and analysis for the MSc thesis on anytime valid inference for probabilistic inflation forecasts across Switzerland (CH), European Union (EU), and United States (US).

## Project Overview

The thesis compares multiple forecasting approaches for inflation prediction.
The analysis evaluates forecast calibration using anytime valid inference methods, including e-values and e-processes for sequential testing of probabilistic forecasts.

## Repository Structure

```
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── R_requirements.R           # R package dependencies
├── .flake8                    # Python linting configuration
├── .gitignore                 # Git ignore patterns
│
├── data/                      # Dataset storage
│   ├── raw/                   # Original, unprocessed data
│   │   ├── ch_data/          # Swiss monetary and inflation data
│   │   ├── eu_data/          # Euro Area economic indicators
│   │   └── us_data/          # US FRED-MD macroeconomic database
│   └── processed/            # Cleaned and transformed datasets
│       ├── *_data_final.csv  # Clean datasets by region
│       ├── *_data_transformed*.csv # Stationary transformed data
│       └── for_drf/          # Design matrices for DRF modeling
│
├── src/                      # Source code modules
│   ├── 01_data_prep/         # Data preprocessing pipeline
│   ├── 02_models_python/     # Python-based models (Baselines, BVAR, DFM)
│   ├── 03_models_r_drf/      # R-based DRF modeling
│   ├── 04_evaluation/        # Analysis and calibration evaluation
│   ├── bvar_utils/           # BVAR implementation utilities
│   └── dfm_utils/            # DFM implementation utilities
│
├── notebooks/                # Jupyter notebooks for exploration
│   ├── 02_*_covariates.ipynb # Covariate analysis by region
│   ├── 02_*_diagnostics.ipynb # Model diagnostics
│   ├── 04_*_inflation.ipynb  # Inflation analysis by region
│   ├── 04_thesis_plots.ipynb # Final thesis visualizations
│   └── helpers.py            # Notebook utility functions
│
└── results/                  # All outputs and analysis results
    ├── forecasts/            # Model prediction outputs
    │   ├── baseline/         # Naive and ARIMA forecasts
    │   ├── bvar/             # BVAR forecasts by prior
    │   ├── dfm/              # Dynamic factor model forecasts
    │   └── drf/              # Distributional random forest forecasts
    ├── plots/                # Generated visualizations
    │   ├── CH/, EU/, US/     # Region-specific plots
    │   └── eda/              # Exploratory data analysis plots
    └── tables/               # Summary statistics and LaTeX tables
```

## Installation and Setup

### 1. Clone Repository
```bash
git clone https://github.com/AmadeoGrob/MSc-thesis.git
cd MSc-thesis
```

### 2. Python Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. R Environment Setup
```bash
# Open R or RStudio and run:
source("R_requirements.R")
```

## Reproduction Workflow

The complete analysis follows this sequential pipeline:

### Step 1: Data Preparation
```bash
cd src/01_data_prep
python data_prep.py
```

This script:
- Processes raw datasets for CH, EU, and US
- Applies transformations for stationarity
- Generates final clean datasets in `data/processed/`
- Creates winsorized versions to handle outliers

### Step 2: Generate Forecasts

#### 2a. Python-based Models (Baselines, BVAR, DFM)
```bash
cd src/02_models_python

# Generate baseline forecasts (Naive, PNC, ARIMA)
# Note: See `02_arima_diagnostics.ipynb` for details on model specification (optional)
python run_baselines.py

# Generate BVAR forecasts with different priors
# Note: See `02_bvar_diagnostics.ipynb` for details on model specification (optional)
python run_bvar.py

# Generate Dynamic Factor Model forecasts
# Note: Need to run `02_dfm_diagnostics.ipynb` first to create `factor_selection.csv` in `results/tables`
python run_dfm.py
```

#### 2b. Prepare Data for R-based DRF Models
```bash
cd src/03_models_r_drf
Rscript data_prep_for_drf.R
```

#### 2c. Generate DRF Forecasts
```bash
cd src/03_models_r_drf
Rscript drf_forecasts.R
```

### Step 3: Evaluation and Analysis
```bash
cd src/04_evaluation
Rscript main_analysis.R
```

This generates:
- Calibration diagnostic tables
- Sequential e-value plots
- Summary statistics
- LaTeX tables

## Key Outputs

### Forecast Files
- `results/forecasts/*/`: CSV files containing point and distributional forecasts

### Analysis Results
- `results/tables/`: Summary statistics and LaTeX tables (partly from notebooks)
- `results/plots/`: Diagnostic plots and visualizations (partly from notebooks)
- Calibration analysis for each model and region

### Tables and Plots
- `*_calibration_summary.tex`: Calibration diagnostic summaries
- Region-specific visualization folders with diagnostic plots

## Notes on runtimes
- `02_arima_diagnostics.ipynb` and `drf_forecasts.R` take a few minutes to run.