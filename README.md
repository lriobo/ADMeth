# ADMeth

**ADMeth** is a pipeline for classification of Illumina methylation array datasets (450k or EPIC) coming from whole-blood samples based on an AI-driven anomaly detection pipeline. Our workflow automatically extracts one feature (REC score) for each one of the 320,000 selected probes. These REC scores identify anomaly levels in each CpG that can be linked to diseases or phenotypic traits.

This pipeline also performs a binary classification task using these features and beta-values for comparison. Graphs, figures and functional and statistical analysis are also included. 

---

## ⚙️ Installation

Make sure you have Python ≥ 3.9.

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows

Install dependencies and ADMeth in editable mode:

    pip install -e .

Then run:

    admeth init 

Before starting to use our repository, you will need to: 

1- Download and unzip the DL model (heavymodelv1) from Zenodo (https://zenodo.org/records/17379350) and place it in models folder (you can change the root in config.yaml).

2- Download and unzip the MSE results for normalization (controlsdatasets) from Zenodo (https://zenodo.org/records/17379350) and place it in data/msemetrics folder (you can change the root in config.yaml).

---
## 📋 Input data structure

Our pipeline requires 2 input datasets: cases and controls, which should be placed in 2 different folders. These datasets should contain Beta-values matrices (CpGs x Samples), coming from Illumina arrays (450k or EPIC) and can be in ".tsv", ".csv" or ".txt" format. The only requirement is that row index must be named ‘CpG’ and contain CpG identifiers. Missing values can be identified as NaNs or as 0.0 values. They should be placed in data/rawdatasets in two separate folders: one for cases and one for controls.

---
## 🔗 Pipeline steps
**0- Initialize**: Creates needed folders and subfolders for the project structure

**1- Preprocess**: Loads raw Beta-values datasets (cases and controls), keeps only selected probes and orders them according to their genomic position.

**2- AD Training** (NOT AVAILABLE YET - SKIPPABLE): Trains an AI anomaly detection model, based on AE, on a whole-blood samples dataset. 

**3- AD Evaluate**: Measures anomalies in terms of MSE in selected datasets (cases and controls).

**4- REC scores**: Obtains a normalized REC score for measuring anomalies.

**5- ML models**: Trains multiple binary classifiers for the selected task with these REC scores (can also be done with raw beta-values for a baseline comparison).

**6- Plots**: Plots classification results in terms of AUC across different ML models and configurations.

**7- Stats**: Plots and tables for REC scores distributions, feature selection, differences in groups and comparison with raw beta-values.

**8- Functional analysis**: Performs an ORA enrichment analysis in KEGG over a selected term to check its correlation with selected features.

**9- Summary**: Generates a .html file with a summary of all the results from previous steps.

---
## 🚀 Usage

Run one of the steps (each one requires previous step):

    admeth init 
    admeth preprocess --config configs/config.yaml
    admeth training --config configs/config.yaml (NOT AVAILABLE YET)
    admeth evaluate --config configs/config.yaml
    admeth recscores --config configs/config.yaml
    admeth mlmodels --config configs/config.yaml
    admeth plots --config configs/config.yaml
    admeth stats --config configs/config.yaml
    admeth functional --config configs/config.yaml
    admeth summary --config configs/config.yaml 

or run all steps:

    admeth all --config configs/config.yaml

---
## 📁 Project structure

    ADMeth/
    ├── configs/                      # configuration files (.yaml)
    ├── src/                          # source code (cli.py, steps_*.py)
    ├── data/                         # data (ignored in Git)
    |   ├────── annotations/           # .csv file needed for the preprocessing
    |   ├───── rawdatasets/           # COPY HERE THE TWO DATASETS OF BETA VALUES (.txt, .csv or .tsv ) FROM CASES AND CONTROLS, ONE IN EACH FOLDER
    |           ├───── cases/
    |           └───── controls/   
    |   ├───── datasets/              # COPY HERE THE UNZIPPED controlsdatasets FOLDER FROM ZENODO (+processed datasets)
    |   ├───── msemetrics/            # mse errors after DL evaluation
    |   ├───── recscores/             # normalized rec scores after measuring anomalies
    |   └───── reports/               # output metrics 
    |           ├───── mlmodels/      # .csv files with the results and plots of the classification task
    |           ├───── stats/         # figures and tables summarizing the statistical analysis of rec scores and betas
    |           ├───── functional/    # figures and .csv files summarizing the functional analysis
    |           └───── summary/       # .html file with the summary of all the results
    ├── models/                       # COPY HERE THE UNZIPPED heavymodelv1 FOLDER FROM ZENODO 
    ├── pyproject.toml                # package metadata
    ├── requirements.txt              # dependencies
    └── README.md   
---
## 🧠 Notes

- Modify `configs/config.yaml` to change dataset paths, configuration parameters or model names.
- Keep large files (data, models, reports) out of GitHub — they’re ignored by `.gitignore`.
- You can add new steps as `steps_*.py` files inside `src/`.
