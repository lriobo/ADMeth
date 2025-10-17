# src/steps_init.py
from pathlib import Path

DEFAULT_DIRS = [
    "configs",
    "src",
    "data/annotations",
    "data/rawdatasets/cases",
    "data/rawdatasets/controls",
    "data/datasets",
    "data/msemetrics",
    "data/recscores",
    "data/reports/mlmodels",
    "data/reports/stats",
    "data/reports/functional",
    "data/reports/summary",
    "models",
]

def run(cfg=None):
    created = []
    for d in DEFAULT_DIRS:
        p = Path(d).resolve()
        p.mkdir(parents=True, exist_ok=True)
        created.append(str(p))
    print("✅ Folders created:")
    for c in created:
        print("  -", c)
    print("\n(Now download, unzip and upload locally the files from Zenodo: heavymodelv1 and controlsdatasets")
