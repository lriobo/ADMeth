# ADMeth

**ADMeth** is a pipeline for classification of methylation data coming from whole-blood samples based on an AI-driven anomaly detection pipeline.

---

## ⚙️ Installation

Make sure you have Python ≥ 3.9.

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows

Install dependencies and ADMeth in editable mode:

    pip install -e .

---

## 🚀 Usage

Run one of the steps (each one requires previous step):

    admeth preprocess --config configs/config.yaml
    admeth training --config configs/config.yaml
    admeth evaluate --config configs/config.yaml
    admeth recscores --config configs/config.yaml
    admeth mlmodels --config configs/config.yaml
    admeth plots --config configs/config.yaml
    admeth stats --config configs/config.yaml
    admeth functional --config configs/config.yaml

or run all steps:

    admeth all --config configs/config.yaml

---

## 📁 Project structure

    AAADMeth/
    ├── configs/           # configuration files (.yaml)
    ├── src/               # source code (cli.py, steps_*.py)
    ├── data/              # data (ignored in Git)
    ├── models/            # trained models (ignored in Git)
    ├── reports/           # output metrics (ignored in Git)
    ├── pyproject.toml     # package metadata
    ├── requirements.txt   # dependencies
    └── README.md          # this file

---

## 🔗 Pipeline steps

1- Preprocess: Loads raw datasets (cases and controls), keeps only selected probes and orders them according to its position.

2- AD Training (SKIPPABLE): Trains an AI anomaly detection model, based on AE, on a whole-blood samples dataset. 

3- AD Evaluate: Measures anomalies in terms of MSE in selected datasets (cases and controls).

4- REC scores: Obtains a normalized REC score for measuring anomalies.

5- ML models: Trains multiple binary classifiers for the selected task (can be also done with raw beta-values for a bsaeline).

6- Plots: Plot classification results across different ML models and configurations.

7- Stats: Plots and tables for REC scores distributions, feature selection, differences in groups and comparison with raw beta-values.

8- Functional analysis: Performs an ORA enrichment analysis in KEGG over a selected term to check its correlation with selected features.

---

## 🧠 Notes

- Modify `configs/config.yaml` to change dataset paths or model names.
- Keep large files (data, models, reports) out of GitHub — they’re ignored by `.gitignore`.
- You can add new steps as `steps_*.py` files inside `src/`.
