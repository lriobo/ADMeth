#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from utils_banner import print_banner
import matplotlib.pyplot as plt

# ========================= helpers =========================

def _load_npy_as_df(npy_path: str) -> pd.DataFrame:
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"{npy_path} is not a 2D matrix (shape {arr.shape})")
    arr = arr.astype(np.float32, copy=False)
    arr[arr == -1.0] = np.nan
    cols = [str(i) for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)

def _list_npys_recursive(root: Path) -> list[str]:
    return [str(p) for p in root.rglob("*.npy")]

def _norm_stem(p: str) -> str:
    s = Path(p).stem
    for suf in ("_scores", "_mse_per_sample_per_position"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    return q

def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def _chunked_hist_density(df: pd.DataFrame, bins=512, x_range=(-1, 3), chunk_cols=2000):
    total = np.zeros(bins, dtype=np.float64)
    edges = np.linspace(x_range[0], x_range[1], bins + 1)
    for s in range(0, df.shape[1], chunk_cols):
        e = min(s + chunk_cols, df.shape[1])
        block = df.iloc[:, s:e].to_numpy(dtype=np.float32, copy=False).ravel()
        block = block[np.isfinite(block)]
        if block.size == 0:
            continue
        cnt, _ = np.histogram(block, bins=edges, density=False)
        total += cnt
    width = (x_range[1] - x_range[0]) / bins
    dens = total / (total.sum() * width) if total.sum() > 0 else total
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, dens

def _boxplot_two_groups(ax, data_ctrl, data_case, title, ylabel=None,
                        color_ctrl="#1f77b4", color_case="#d62728"):
    bp = ax.boxplot([data_ctrl, data_case],
                    positions=[1, 2], widths=0.7, showfliers=False, patch_artist=True)
    # colorear cajas
    for patch, c in zip(bp["boxes"], [color_ctrl, color_case]):
        patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor("black")
    # medianas en negro
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.4)
    # whiskers y caps en gris
    for w in bp["whiskers"]:
        w.set_color("gray"); w.set_linewidth(1.2)
    for c in bp["caps"]:
        c.set_color("gray"); c.set_linewidth(1.2)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Controls", "Cases"])
    ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)

# ========================= main =========================

def run(cfg):
    # Banner
    project = str(cfg.get("run", {}).get("project", "default"))
    print_banner(step="stats", project=project)

    paths = cfg.get("paths", {})
    ds_root = Path(paths["datasets_root"]).resolve()                  
    rec_root = Path(paths["recscores"]).resolve() / project           
    out_root = Path(paths.get("reports", "data/reports")).resolve() / "stats" / project
    out_root.mkdir(parents=True, exist_ok=True)

    run_cfg = cfg.get("run", {})
    select = run_cfg.get("select", {})
    sel_cases = list(select.get("cases", []))
    sel_controls = list(select.get("controls", []))

    rec_cases_dir = rec_root / "cases"
    rec_ctrls_dir = rec_root / "controls"
    rec_cases_npys = _list_npys_recursive(rec_cases_dir) if rec_cases_dir.exists() else []
    rec_ctrls_npys = _list_npys_recursive(rec_ctrls_dir) if rec_ctrls_dir.exists() else []
    if sel_cases:
        rec_cases_npys = [p for p in rec_cases_npys if _norm_stem(p) in sel_cases]
    if sel_controls:
        rec_ctrls_npys = [p for p in rec_ctrls_npys if _norm_stem(p) in sel_controls]

    if not rec_cases_npys or not rec_ctrls_npys:
        raise RuntimeError(
            f"[stats] Not REC .npy files found in {rec_cases_dir} o {rec_ctrls_dir} "
            f"for the selected stems. Check if you have ran 'recscores'."
        )


    rec_cases_path = rec_cases_npys[0]
    rec_ctrls_path = rec_ctrls_npys[0]
    print(f"[stats] REC cases:    {rec_cases_path}")
    print(f"[stats] REC controls: {rec_ctrls_path}")

    df_cases_rec = _load_npy_as_df(rec_cases_path)
    df_ctrls_rec = _load_npy_as_df(rec_ctrls_path)
    if list(df_cases_rec.columns) != list(df_ctrls_rec.columns):
        common = [c for c in df_cases_rec.columns if c in df_ctrls_rec.columns]
        if not common:
            raise RuntimeError("[stats] Cases and Controls do not share columns in REC.")
        df_cases_rec = df_cases_rec[common]
        df_ctrls_rec = df_ctrls_rec[common]

    ref_cols = df_cases_rec.columns  

    raw_cases_df = None
    raw_ctrls_df = None
    if sel_cases and sel_controls:
        raw_cases_path = ds_root / "cases" / f"{sel_cases[0]}.npy"
        raw_ctrls_path = ds_root / "controls" / f"{sel_controls[0]}.npy"
        if raw_cases_path.exists() and raw_ctrls_path.exists():
            def _load_raw_align(p: Path, ref_cols: pd.Index) -> pd.DataFrame:
                arr = np.load(p).astype(np.float32, copy=False)
                if arr.ndim != 2:
                    raise ValueError(f"{p} is not a 2D matrix (shape {arr.shape})")
                n_features = len(ref_cols)
                if arr.shape[1] == n_features:
                    pass  # (n_samples, n_features)
                elif arr.shape[0] == n_features:
                    arr = arr.T  # (n_features, n_samples) -> (n_samples, n_features)
                else:
                    raise ValueError(f"{p} doues not fit n_features={n_features}. Shape: {arr.shape}")
                return pd.DataFrame(arr, columns=ref_cols)

            raw_cases_df = _load_raw_align(raw_cases_path, ref_cols)
            raw_ctrls_df = _load_raw_align(raw_ctrls_path, ref_cols)
            print(f"[stats] RAW cases:    {raw_cases_path}")
            print(f"[stats] RAW controls: {raw_ctrls_path}")
        else:
            print("[stats] (RAW opcional) Not .npy RAW found for the selected stems.")

    # === TOP FEATURES (KS) — REC ===
    pvals_rec = []
    n_feats = len(df_cases_rec.columns)
    for i, col in enumerate(df_cases_rec.columns, start=1):
        x = df_cases_rec[col].to_numpy()
        y = df_ctrls_rec[col].to_numpy()
        x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            p = np.nan
        else:
            try:
                p = ks_2samp(x, y, alternative="two-sided", mode="auto").pvalue
            except Exception:
                p = np.nan
        pvals_rec.append((col, p))
        if (i % 32000 == 0) or (i == n_feats):
            print(f"[stats][REC] KS per-feature: {i}/{n_feats} columns")

    tf_rec = pd.DataFrame(pvals_rec, columns=["feature", "p_ks"])
    tf_rec["p_ks"] = tf_rec["p_ks"].astype(float)
    tf_rec["fdr_bh"] = _bh_fdr(tf_rec["p_ks"].fillna(1.0).to_numpy())
    tf_rec = tf_rec.sort_values("p_ks", ascending=True, na_position="last").reset_index(drop=True)
    top_csv_rec = out_root / "top_features_ks_rec.csv"
    tf_rec.to_csv(top_csv_rec, index=False)
    print(f"[stats] KS test results per feature saved in (KS, REC): {top_csv_rec}")

    # === TOP FEATURES (KS) — RAW / Beta-values ===
    if (raw_cases_df is not None) and (raw_ctrls_df is not None):
        pvals_beta = []
        n_feats_b = len(ref_cols)
        for i, col in enumerate(ref_cols, start=1):
            x = raw_cases_df[col].to_numpy()
            y = raw_ctrls_df[col].to_numpy()
            x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
            if x.size == 0 or y.size == 0:
                p = np.nan
            else:
                try:
                    p = ks_2samp(x, y, alternative="two-sided", mode="auto").pvalue
                except Exception:
                    p = np.nan
            pvals_beta.append((col, p))
            if (i % 32000 == 0) or (i == n_feats_b):
                print(f"[stats][Beta] KS per-feature: {i}/{n_feats_b} columns")

        tf_beta = pd.DataFrame(pvals_beta, columns=["feature", "p_ks"])
        tf_beta["p_ks"] = tf_beta["p_ks"].astype(float)
        tf_beta["fdr_bh"] = _bh_fdr(tf_beta["p_ks"].fillna(1.0).to_numpy())
        tf_beta = tf_beta.sort_values("p_ks", ascending=True, na_position="last").reset_index(drop=True)
        top_csv_beta = out_root / "top_features_ks_beta.csv"
        tf_beta.to_csv(top_csv_beta, index=False)
        print(f"[stats] KS test results per feature saved in (KS, Beta-values): {top_csv_beta}")
    else:
        tf_beta = None

    # === PLOTS (TRAIN only) ===
    plots_dir = out_root / "plots"
    feats_dir = plots_dir / "features"
    plots_dir.mkdir(parents=True, exist_ok=True)
    feats_dir.mkdir(parents=True, exist_ok=True)

    x_c, d_c = _chunked_hist_density(df_cases_rec, bins=512, x_range=(-1, 3))
    x_n, d_n = _chunked_hist_density(df_ctrls_rec, bins=512, x_range=(-1, 3))
    eps = 1e-12
    d_c = np.clip(d_c, eps, None); d_n = np.clip(d_n, eps, None)

    plt.figure(figsize=(9, 5))
    plt.plot(x_c, d_c, label="TRAIN Cases (REC)")
    plt.plot(x_n, d_n, label="TRAIN Controls (REC)")
    plt.yscale("log"); plt.xlim(-1, 3)
    plt.xlabel("REC value"); plt.ylabel("Density (log)")
    plt.title("Global density –  REC | Cases vs Controls")
    plt.legend()
    _savefig(plots_dir / "global_density_TRAIN_REC_logY.png")

    med_cases = df_cases_rec.median(axis=0).to_numpy()
    med_ctrls = df_ctrls_rec.median(axis=0).to_numpy()
    max_pts = 100_000
    n = med_cases.size
    if n > max_pts:
        idx = np.random.default_rng(42).choice(n, size=max_pts, replace=False)
        xs, ys = med_cases[idx], med_ctrls[idx]
    else:
        xs, ys = med_cases, med_ctrls

    plt.figure(figsize=(7.6, 7.0))
    plt.scatter(xs, ys, s=8, alpha=0.6)
    mn = float(min(xs.min(), ys.min())); mx = float(max(xs.max(), ys.max()))
    plt.plot([mn, mx], [mn, mx], color="black", linewidth=1)
    plt.xscale("symlog", linthresh=1e-3); plt.yscale("symlog", linthresh=1e-3)
    plt.xlabel("Median per feature (Cases, REC)")
    plt.ylabel("Median per feature (Controls, REC)")
    plt.title("Per-feature medians REC — symlog axes")
    _savefig(plots_dir / "scatter_medians_TRAIN_REC_symlog.png")

    TOPK_BOX = 10
    top_feats = tf_rec["feature"].astype(str).head(TOPK_BOX).tolist()

    def _cap_series(a: np.ndarray, cap: int = 12000):
        a = pd.Series(a).dropna()
        if a.shape[0] > cap:
            return a.sample(cap, random_state=42).to_numpy()
        return a.to_numpy()

    for f in top_feats:
        # REC
        c_rec = _cap_series(df_cases_rec[f].to_numpy())
        n_rec = _cap_series(df_ctrls_rec[f].to_numpy())

        if (raw_cases_df is not None) and (raw_ctrls_df is not None):
            # Beta-values (RAW) vs REC
            plt.figure(figsize=(12, 5.2))
            ax1 = plt.subplot(1, 2, 1)
            raw_ctrl_vals = _cap_series(raw_ctrls_df[f].to_numpy())
            raw_case_vals = _cap_series(raw_cases_df[f].to_numpy())
            _boxplot_two_groups(ax1, raw_ctrl_vals, raw_case_vals,
                                title=f"Beta-values — feature {f}", ylabel="Value",
                                color_ctrl="#1f77b4", color_case="#d62728")

            ax2 = plt.subplot(1, 2, 2)
            _boxplot_two_groups(ax2, n_rec, c_rec,
                                title=f"REC — feature {f}", ylabel=None,
                                color_ctrl="#1f77b4", color_case="#d62728")

            _savefig(feats_dir / f"feature_{f}_box_Beta_vs_REC.png")
        else:
            # Solo REC
            plt.figure(figsize=(6.5, 5.0))
            ax = plt.gca()
            _boxplot_two_groups(ax, n_rec, c_rec,
                                title=f"REC — feature {f}", ylabel=None,
                                color_ctrl="#1f77b4", color_case="#d62728")
            _savefig(feats_dir / f"feature_{f}_box_REC.png")

    print(f"✅ [stats] Finished. Outputs saved in: {out_root}")

