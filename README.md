# ADMeth

**ADMeth** is a command-line pipeline for methylation model evaluation.

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

Run the evaluation pipeline:

    admeth evaluate --config configs/config.yaml

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

## 🧠 Notes

- Modify `configs/config.yaml` to change dataset paths or model names.
- Keep large files (data, models, reports) out of GitHub — they’re ignored by `.gitignore`.
- You can add new steps as `steps_*.py` files inside `src/`.
