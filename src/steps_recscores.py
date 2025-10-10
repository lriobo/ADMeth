#!/usr/bin/env python
# coding: utf-8

import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from utils_banner import print_banner


def run(cfg):
    project = str(cfg.get("run", {}).get("project", "default"))
    print_banner(step="recscores", project=project)
    print("###############   04 - REC SCORES   ###############")
    run_cfg = cfg.get("run", {})
    project = str(run_cfg.get("project", "default"))

    mse_root = (Path(cfg["paths"]["msemetrics"]).resolve() / project)
    out_root = (Path(cfg["paths"]["recscores"]).resolve() / project)
    out_root.mkdir(parents=True, exist_ok=True)
    
    cases_dir         = (mse_root / "cases").resolve()
    proc_controls_dir = (mse_root / "controls").resolve()     
    norm_controls_dir = Path(cfg["paths"].get("norm_controls_root", proc_controls_dir)).resolve()

    print(f"Normalization controls from: {norm_controls_dir}")
    print(f"Processing cases:   {cases_dir}")
    print(f"Processing controls:    {proc_controls_dir}")
    print(f"Processing outputs:         {out_root}")

    rs_cfg = cfg.get("recscores", {})
    group_size    = int(rs_cfg.get("group_size", 10))   
    missing_value = float(rs_cfg.get("missing_value", -1))

    def group_columns_by_mean(data, group_size: int = 10, missing_value: float = -1):

        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise TypeError("Input should be a pandas.DataFrame or a numpy.ndarray.")

        n_cols = df.shape[1]

        if group_size == 1:
            grouped = df.mask(df.eq(missing_value), np.nan)
            return grouped.fillna(missing_value)

        grouped_cols = []
        for start in range(0, n_cols, group_size):
            end = min(start + group_size, n_cols)
            subset = df.iloc[:, start:end]
            group_mean = subset.mask(subset.eq(missing_value), np.nan).mean(axis=1, skipna=True)
            group_mean = group_mean.fillna(missing_value)
            grouped_cols.append(group_mean)

        grouped_matrix = pd.concat(grouped_cols, axis=1)
        return grouped_matrix

    def background_normalization(mat: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(mat.astype(np.float32, copy=False))
        med = df.median(axis=1).replace(0, np.nan)
        df = (df.sub(med, axis=0)).div(med, axis=0)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df


    def score_region(bc_errors: pd.DataFrame, ctrl_mean: pd.Series, ctrl_std: pd.Series) -> pd.DataFrame:
        z = (bc_errors - ctrl_mean) / (80+ctrl_std.replace(0, np.nan))
        return z.fillna(0.0)

    norm_control_files = sorted(norm_controls_dir.rglob("*.npy"))
    case_files         = sorted(cases_dir.rglob("*.npy"))
    proc_control_files = sorted(proc_controls_dir.rglob("*.npy"))

    if not norm_control_files:
        raise RuntimeError(f"No .npy file found in: {norm_controls_dir}")
    if not case_files and not proc_control_files:
        raise RuntimeError(f"No .npy file found in: {cases_dir} a/o {proc_controls_dir}")

    bc_controls_list = []
    for f in norm_control_files:
        mat = np.load(f)  # (n_samples, n_features)
        grouped = group_columns_by_mean(mat, group_size=group_size, missing_value=missing_value)
        bc_controls_list.append(background_normalization(grouped.values))

    BCTrainConMSE = pd.concat(bc_controls_list, axis=0, ignore_index=True)
    ctrl_mean = BCTrainConMSE.mean(axis=0)
    ctrl_std  = BCTrainConMSE.std(axis=0).replace(0, np.nan)
    ctrl_std  = ctrl_std.fillna(1e-8)

    for gname, files in (("controls", proc_control_files), ("cases", case_files)):
        g_out = (out_root / gname)
        g_out.mkdir(parents=True, exist_ok=True)

        for f in files:
            dataset = f.parent.name
            if dataset in ("cases", "controls"):
                dataset = f.stem.replace("_mse_per_sample_per_position", "")
            print(f"[{gname}] processing {dataset}...")
        
            z = None  
            try:
                mat = np.load(f).astype(np.float32, copy=False)  
                grouped = group_columns_by_mean(mat, group_size=group_size, missing_value=missing_value)
                bc = background_normalization(grouped.values)
                z  = (bc - ctrl_mean) / (ctrl_std)  
                z  = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            except Exception as e:
                print(f"[{gname}] ERROR: fail processing {f.name}: {e}")
                continue
        
            if z is None or z.empty:
                print(f"[{gname}] WARNING: {dataset} without valid matrix, omitting...")
                continue
        
            out_npy = (out_root / gname / f"{dataset}_scores.npy")
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, z.to_numpy(dtype=np.float32))
            print(f"[{gname}] OK — saved {out_npy}")

    print("Finished")
