#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import numpy as np
import pandas as pd
from utils_banner import print_banner

ALLOWED_EXTS = {".csv", ".tsv", ".txt"}

def _read_table_auto(fp: Path) -> pd.DataFrame:
    """Lee CSV/TSV intentando detectar separador y comprobando columna CpG."""
    # prueba separadores comunes
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
        raise ValueError(f"{fp} no tiene columna 'CpG'.")
    return df

def _dedup_cpg(df: pd.DataFrame, how: str = "first") -> pd.DataFrame:
    """Resuelve CpGs duplicadas (first | mean | median)."""
    df = df.copy()
    if how == "first":
        return df.drop_duplicates(subset=["CpG"], keep="first")
    agg = {"mean": "mean", "median": "median"}.get(how.lower(), "mean")
    return df.groupby("CpG", as_index=False).agg(agg)

def _filter_complete_align(df: pd.DataFrame, anno_cpg_index: pd.Index) -> pd.DataFrame:
    """
    Entrada: df con columna 'CpG' y columnas de muestras (CpG x muestras).
    Salida: DataFrame indexado por anno_cpg_index, columnas = muestras, sin NaN (rellenos a 0).
    """
    df = df.copy()
    df["CpG"] = df["CpG"].astype(str).str.strip()
    df = df.set_index("CpG")
    # quedarnos con CpGs del annotation y reordenarlas
    df = df.loc[df.index.intersection(anno_cpg_index)]
    df = df.reindex(index=anno_cpg_index)
    # Missings SIEMPRE a 0
    df = df.fillna(0).astype(np.float32, copy=False)
    # sanity prints
    vals = df.to_numpy()
    print(f"    ↳ Max value: {np.nanmax(vals):.6g}")
    print(f"    ↳ Min value: {np.nanmin(vals):.6g}")
    return df

def _save_npy_matrix(df_aligned: pd.DataFrame, out_path: Path):
    """
    Guarda como .npy con shape (n_samples, n_features).
    df_aligned está en (n_features, n_samples) -> trasponemos.
    """
    arr = df_aligned.to_numpy(dtype=np.float32, copy=False).T
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    print(f"    ↳ Saved: {out_path}  shape={arr.shape} (samples, features)")

def _find_by_stem_recursive(root: Path, stem: str) -> Path:
    """
    Busca recursivamente en 'root' un archivo cuyo nombre base (sin extensión) == stem
    y extensión en ALLOWED_EXTS. Devuelve el primero por orden alfabético.
    """
    cand = sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in ALLOWED_EXTS and p.stem == stem)
    if not cand:
        raise FileNotFoundError(f"No se encontró un fichero con stem '{stem}' en {root}")
    return cand[0]

def _process_one(fp: Path, out_dir: Path, anno_cpg_index: pd.Index, dedup: str = "first"):
    print(f"[preprocess] Reading: {fp.relative_to(fp.anchor) if fp.is_absolute() else fp}")
    df = _read_table_auto(fp)
    if "CpG" not in df.columns:
        raise ValueError(f"{fp} no tiene columna 'CpG'.")

    if df["CpG"].duplicated().any():
        print(f"    ↳ Duplicates in CpG detected. Resolving with: {dedup}")
        df = _dedup_cpg(df, how=dedup)

    df_aligned = _filter_complete_align(df, anno_cpg_index)
    out_path = out_dir / (fp.stem + ".npy")
    _save_npy_matrix(df_aligned, out_path)

    # opcional: guardar orden de CpGs y nombres de muestras
    (out_dir / f"{fp.stem}_cpg_order.txt").write_text("\n".join(anno_cpg_index.tolist()))
    (out_dir / f"{fp.stem}_samples.txt").write_text("\n".join(df_aligned.columns.astype(str).tolist()))

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
        raise RuntimeError("Debes definir run.select.cases y run.select.controls con los stems a usar.")

    # cargar annotation y preparar índice
    print(f"[preprocess] Loading annotation: {anno_fp}")
    anno = pd.read_csv(anno_fp)
    if "CpG" not in anno.columns:
        raise ValueError("El archivo de anotaciones no tiene columna 'CpG'.")
    anno_cpg = anno["CpG"].astype(str).str.strip()
    anno_cpg = anno_cpg[~anno_cpg.duplicated()].reset_index(drop=True)
    anno_cpg_index = pd.Index(anno_cpg, name="CpG")

    # salidas
    out_cases_dir    = ds_root / "cases"
    out_controls_dir = ds_root / "controls"

    # procesar CASES
    print(f"[preprocess] Searching raw CASES in: {raw_root}")
    for stem in stems_cases:
        fp = _find_by_stem_recursive(raw_root, stem)
        _process_one(fp, out_cases_dir, anno_cpg_index, dedup="first")

    # procesar CONTROLS
    print(f"[preprocess] Searching raw CONTROLS in: {raw_root}")
    for stem in stems_controls:
        fp = _find_by_stem_recursive(raw_root, stem)
        _process_one(fp, out_controls_dir, anno_cpg_index, dedup="first")

    print("✅ [preprocess] Completado. Betas .npy listos en datasets_root/cases & datasets_root/controls")
