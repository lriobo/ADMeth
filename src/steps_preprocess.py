#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import numpy as np
import pandas as pd
from utils_banner import print_banner

ALLOWED_EXTS = {".csv", ".tsv", ".txt"}

def _read_table_auto(fp: Path) -> pd.DataFrame:
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(fp, sep=sep)
            if "CpG" in df.columns:
                return df
        except Exception:
            pass
    # fallback simple
    df = pd.read_csv(fp)
    if "CpG" not in df.columns:
        raise ValueError(f"{fp} missing 'CpG' column.")
    return df

def _dedup_cpg(df: pd.DataFrame, how: str = "first") -> pd.DataFrame:
    df = df.copy()
    if how == "first":
        return df.drop_duplicates(subset=["CpG"], keep="first")
    agg = {"mean": "mean", "median": "median"}.get(how.lower(), "mean")
    return df.groupby("CpG", as_index=False).agg(agg)

def _filter_complete_align(df: pd.DataFrame, anno_cpg_index: pd.Index) -> pd.DataFrame:

    df = df.copy()
    df["CpG"] = df["CpG"].astype(str).str.strip()
    df = df.set_index("CpG")
    
    df = df.loc[df.index.intersection(anno_cpg_index)]
    df = df.reindex(index=anno_cpg_index)
    
    df = df.fillna(0).astype(np.float32, copy=False)
    
    vals = df.to_numpy()
    print(f"    ↳ Max value: {np.nanmax(vals):.6g}")
    print(f"    ↳ Min value: {np.nanmin(vals):.6g}")
    return df

def _save_npy_matrix(df_aligned: pd.DataFrame, out_path: Path):

    arr = df_aligned.to_numpy(dtype=np.float32, copy=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    print(f"    ↳ Saved: {out_path}  shape={arr.shape} (samples, features)")

def _find_by_stem_recursive(root: Path, stem: str) -> Path:
    cand = sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in ALLOWED_EXTS and p.stem == stem)
    if not cand:
        raise FileNotFoundError(f"Could not find a file with stem '{stem}' en {root}")
    return cand[0]

def _process_one(fp: Path, out_dir: Path, anno_cpg_index: pd.Index, dedup: str = "first"):
    print(f"[preprocess] Reading: {fp.relative_to(fp.anchor) if fp.is_absolute() else fp}")
    df = _read_table_auto(fp)
    if "CpG" not in df.columns:
        raise ValueError(f"{fp} missing 'CpG' column.")

    if df["CpG"].duplicated().any():
        print(f"    ↳ Duplicates in CpG detected. Resolving with: {dedup}")
        df = _dedup_cpg(df, how=dedup)

    df_aligned = _filter_complete_align(df, anno_cpg_index)
    out_path = out_dir / (fp.stem + ".npy")
    _save_npy_matrix(df_aligned, out_path)

    #(out_dir / f"{fp.stem}_cpg_order.txt").write_text("\n".join(anno_cpg_index.tolist()))
    #(out_dir / f"{fp.stem}_samples.txt").write_text("\n".join(df_aligned.columns.astype(str).tolist()))

def run(cfg):
    project = str(cfg.get("run", {}).get("project", "default"))
    print_banner(step="preprocess", project=project)

    paths = cfg.get("paths", {})
    anno_fp = Path(paths["annotations"]).resolve()
    raw_root = Path(paths["raw_datasets_root"]).resolve()
    ds_root  = Path(paths["datasets_root"]).resolve()

    select = cfg.get("run", {}).get("select", {})
    stems_cases    = list(select.get("cases", []))
    stems_controls = list(select.get("controls", []))

    if not stems_cases or not stems_controls:
        raise RuntimeError("You must define run.select.cases and run.select.controls with the stems to use.")

    print(f"[preprocess] Loading annotation: {anno_fp}")
    anno = pd.read_csv(anno_fp)
    if "CpG" not in anno.columns:
        raise ValueError("Anno file missing 'CpG' column.")
    anno_cpg = anno["CpG"].astype(str).str.strip()
    anno_cpg = anno_cpg[~anno_cpg.duplicated()].reset_index(drop=True)
    anno_cpg_index = pd.Index(anno_cpg, name="CpG")

    out_cases_dir    = ds_root / "cases"
    out_controls_dir = ds_root / "controls"

    print(f"[preprocess] Searching raw CASES in: {raw_root}")
    for stem in stems_cases:
        fp = _find_by_stem_recursive(raw_root, stem)
        _process_one(fp, out_cases_dir, anno_cpg_index, dedup="first")

    print(f"[preprocess] Searching raw CONTROLS in: {raw_root}")
    for stem in stems_controls:
        fp = _find_by_stem_recursive(raw_root, stem)
        _process_one(fp, out_controls_dir, anno_cpg_index, dedup="first")

    print("✅ [preprocess] Finished. Betas .npy ready in datasets_root/cases & datasets_root/controls")
