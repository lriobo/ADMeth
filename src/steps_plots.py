# src/steps_plots.py
#!/usr/bin/env python
# coding: utf-8

import os, glob, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from pathlib import Path

# ---- fixed display ranges ----
HEATMAP_VMIN = 0.3
HEATMAP_VMAX = 0.8
LINE_YLIM   = (0.2, 0.8)
BOX_YLIM    = (0.3, 0.8)
Y_TICK_STEP_HEATMAP = 30   # k ticks every 30

# ---- simple logger for report ----
class _Logger:
    def __init__(self): self.lines = []
    def log(self, msg): print(msg); self.lines.append(str(msg))
    def html(self): return "<pre style='white-space:pre-wrap; font-size:12px;'>" + "\n".join(self.lines) + "</pre>"

# ---- detect header and prefix from folder name ----
def _derive_header_and_prefix(results_dir: str):
    base = os.path.basename(os.path.normpath(results_dir))
    low  = base.lower()
    header = None

    # BetasAll -> "All Cases vs Controls"
    if low.startswith("betas"):
        header = "Betas: Cases vs Controls"
    elif low.startswith("rec"):
        header = "REC: Cases vs Controls"
    if header is None:
        header = base

    prefix = base.replace(" ", "_")
    return header, prefix

# ---- file discovery ----
def _ignore_output_csv(name_lower: str) -> bool:
    ignore_keywords = [
        "combined_performance_long", "summary_table", "auc_vs_k_with_ci",
        "auc_heatmap", "auc_boxplot_per_model",
        "mean_auc_per_model", "top20_by_model_k"
    ]
    return any(kw in name_lower for kw in ignore_keywords)

def _find_csvs(results_dir: str):
    all_csvs = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    main_csvs, bootstrap_csvs = [], []
    for fp in all_csvs:
        name_lower = os.path.basename(fp).lower()
        if _ignore_output_csv(name_lower):
            continue
        if name_lower.endswith("_bootstrap.csv"):
            bootstrap_csvs.append(fp)
        elif name_lower.endswith("_top20_features.csv"):
            continue
        else:
            try:
                head = pd.read_csv(fp, nrows=2)
                if "k" in head.columns and head.shape[1] >= 2:
                    main_csvs.append(fp)
            except Exception:
                pass
    return main_csvs, bootstrap_csvs

# ---- load & merge ----
def _load_main(main_csvs):
    dfs = []
    for fp in main_csvs:
        df = pd.read_csv(fp)
        df_long = df.melt(id_vars=["k"], var_name="model", value_name="auc_main")
        df_long["source_file"] = os.path.basename(fp)
        dfs.append(df_long)
    main_all = pd.concat(dfs, ignore_index=True)
    main_agg = (main_all.groupby(["k","model"], as_index=False)
                        .agg(auc_main=("auc_main","mean")))
    return main_agg

def _load_bootstrap(bootstrap_csvs):
    if not bootstrap_csvs:
        return None
    dfs = []
    for fp in bootstrap_csvs:
        try:
            dfb = pd.read_csv(fp)
            expected = {"k","model","mean_auc_boot","ci95_low","ci95_high"}
            if expected.issubset(set(dfb.columns)):
                dfb["source_file"] = os.path.basename(fp)
                dfs.append(dfb[["k","model","mean_auc_boot","ci95_low","ci95_high","source_file"]])
        except Exception:
            pass
    if not dfs:
        return None
    bs_all = pd.concat(dfs, ignore_index=True)
    bs_agg = (bs_all.groupby(["k","model"], as_index=False)
                    .agg(mean_auc_boot=("mean_auc_boot","mean"),
                         ci95_low=("ci95_low","mean"),
                         ci95_high=("ci95_high","mean")))
    return bs_agg

def _merge_perf(main_agg, bs_agg):
    if bs_agg is not None:
        perf = pd.merge(main_agg, bs_agg, on=["k","model"], how="left")
        perf["auc_plot"] = perf["mean_auc_boot"].fillna(perf["auc_main"])
    else:
        perf = main_agg.copy()
        perf["mean_auc_boot"] = np.nan
        perf["ci95_low"] = np.nan
        perf["ci95_high"] = np.nan
        perf["auc_plot"] = perf["auc_main"]
    return perf

# ---- table -> image ----
def _save_table_as_image(values_df: pd.DataFrame, out_path: str,
                         cmap_name: str = "coolwarm",
                         vmin: float = HEATMAP_VMIN, vmax: float = HEATMAP_VMAX,
                         header_text: str = None):
    data = values_df.values.astype(float)
    fig_h = max(3, 0.5 * (data.shape[0] + 2))
    fig_w = max(4, 1.0 * (data.shape[1] + 2))
    plt.figure(figsize=(fig_w, fig_h))
    if header_text:
        plt.suptitle(header_text, y=0.99, fontsize=12, fontweight="bold")
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(data, aspect="auto", norm=norm, cmap=cmap_name)
    plt.colorbar(label="AUC")
    ax = plt.gca()
    ax.grid(True, which='both', axis='both', alpha=0.15)
    plt.xticks(np.arange(data.shape[1]), values_df.columns, rotation=45, ha="right")
    plt.yticks(np.arange(data.shape[0]), values_df.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---- plots ----
def _plot_auc_vs_k(perf, out_path, header_text=None):
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    if header_text:
        plt.suptitle(header_text, y=0.99, fontsize=12, fontweight="bold")
    for model, dfm in perf.groupby("model"):
        dfm = dfm.sort_values("k")
        plt.plot(dfm["k"], dfm["auc_plot"], label=str(model))
        mask = dfm["ci95_low"].notna() & dfm["ci95_high"].notna()
        if mask.any():
            plt.fill_between(dfm.loc[mask, "k"],
                             dfm.loc[mask, "ci95_low"],
                             dfm.loc[mask, "ci95_high"],
                             alpha=0.18)
    plt.xlabel("Number of features (k)")
    plt.ylabel("AUC (with 95% CI)")
    plt.title("Model performance across k")
    ax.set_ylim(*LINE_YLIM)
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1)
    ax.set_xlim(0, 500)
    ax.margins(x=0)
    for sp in ["left","bottom","right","top"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(1.3)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.xaxis.grid(False)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_heatmap(perf, out_path, header_text=None):
    pivot = perf.pivot_table(index="k", columns="model", values="auc_plot", aggfunc="mean")
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    plt.figure(figsize=(10, 7))
    if header_text:
        plt.suptitle(header_text, y=0.99, fontsize=12, fontweight="bold")
    im = plt.imshow(pivot.values, aspect="auto", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    plt.colorbar(im, label="AUC")
    ks = pivot.index.values
    models = pivot.columns.tolist()
    if len(ks) > 0:
        try:
            ks_int = ks.astype(int)
            tick_positions = np.where((ks_int % Y_TICK_STEP_HEATMAP) == 0)[0]
            tick_labels = ks[tick_positions]
        except Exception:
            tick_positions = np.arange(0, len(ks), Y_TICK_STEP_HEATMAP)
            tick_labels = ks[tick_positions]
        if len(tick_positions) == 0:
            tick_positions = np.arange(0, len(ks), max(1, len(ks)//10))
            tick_labels = ks[tick_positions]
        plt.yticks(tick_positions, tick_labels)
    plt.xticks(np.arange(len(models)), models, rotation=45, ha="right")
    plt.ylabel("k (number of features)")
    plt.title("AUC heatmap across k and models")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_boxplot_per_model(perf, out_path, header_text=None):
    models = sorted(perf["model"].unique().tolist())
    data = []
    for m in models:
        vals = perf.loc[perf["model"] == m, "auc_plot"].dropna().values
        if vals.size == 0:
            vals = np.array([np.nan])
        data.append(vals)
    plt.figure(figsize=(10, 6))
    if header_text:
        plt.suptitle(header_text, y=0.99, fontsize=12, fontweight="bold")
    plt.boxplot(data, labels=models, showfliers=False)
    ax = plt.gca()
    ax.set_ylim(*BOX_YLIM)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("AUC across k")
    plt.title("AUC distribution per model (boxplot across k)")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.xaxis.grid(False)
    ax.axhline(0.5, linestyle="--", color="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---- Top-20 CSV (model, k) ----
def _save_top20_model_k(perf, out_csv):
    top = perf.sort_values("auc_plot", ascending=False).loc[:, ["model", "k", "auc_plot", "ci95_low", "ci95_high"]]
    top20 = top.head(20).reset_index(drop=True)
    top20.to_csv(out_csv, index=False)
    return top20

# ---- helpers for HTML report ----
def _img_to_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext in [".png"] else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def _write_report(report_path, logger_html, images, tables_html, header_text):
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>CV Results Report</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} h1{margin-bottom:6px;} h2{margin-top:24px;} img{max-width:100%;height:auto;} .section{margin-bottom:24px;} </style>",
        "</head><body>",
        f"<h1>{header_text}</h1>",
        "<div class='section'><h2>Run log</h2>",
        logger_html,
        "</div>"
    ]
    for title, path in images:
        parts.append("<div class='section'>")
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<img src='{_img_to_data_uri(path)}' alt='{title}'>")
        parts.append("</div>")
    for title, html in tables_html:
        parts.append("<div class='section'>")
        parts.append(f"<h2>{title}</h2>")
        parts.append(html)
        parts.append("</div>")
    parts.append("</body></html>")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

# ---- main entry (callable) ----
def visualize_cv_results(results_dir: str, save_outputs: bool = True, make_html_report: bool = True):
    logger = _Logger()
    if not os.path.isdir(results_dir):
        logger.log(f"[Info] Directory not found: {results_dir}")
        return

    header_text, prefix = _derive_header_and_prefix(results_dir)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    main_csvs, bootstrap_csvs = _find_csvs(results_dir)
    if not main_csvs:
        logger.log("[Info] No per-k CSV files found. Expected files with a 'k' column and model columns.")
        return

    logger.log(f"Found {len(main_csvs)} per-k CSV(s) and {len(bootstrap_csvs)} bootstrap CSV(s).")
    logger.log(f"Header: {header_text}")
    logger.log(f"Prefix: {prefix}")

    main_agg = _load_main(main_csvs)
    bs_agg = _load_bootstrap(bootstrap_csvs)
    perf = _merge_perf(main_agg, bs_agg)

    combined_perf_path = os.path.join(plots_dir, f"{prefix}_combined_performance_long.csv")
    perf.to_csv(combined_perf_path, index=False)
    logger.log(f"Saved: {combined_perf_path}")

    line_path = os.path.join(plots_dir, f"{prefix}_auc_vs_k_with_ci.png")
    _plot_auc_vs_k(perf, line_path, header_text=header_text)
    logger.log(f"Saved: {line_path}")

    heatmap_path = os.path.join(plots_dir, f"{prefix}_auc_heatmap.png")
    _plot_heatmap(perf, heatmap_path, header_text=header_text)
    logger.log(f"Saved: {heatmap_path}")

    boxplot_path = os.path.join(plots_dir, f"{prefix}_auc_boxplot_per_model.png")
    _plot_boxplot_per_model(perf, boxplot_path, header_text=header_text)
    logger.log(f"Saved: {boxplot_path}")

    summary_rows = []
    for model, dfm in perf.groupby("model"):
        if dfm.empty:
            continue
        best_idx = dfm["auc_plot"].idxmax()
        summary_rows.append({
            "model": model,
            "best_auc": float(dfm.loc[best_idx, "auc_plot"]),
            "mean_auc": float(dfm["auc_plot"].mean())
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("best_auc", ascending=False).reset_index(drop=True)
    models = summary_df["model"].tolist()
    table_df = pd.DataFrame({
        m: [summary_df.loc[summary_df["model"]==m, "best_auc"].values[0],
            summary_df.loc[summary_df["model"]==m, "mean_auc"].values[0]]
        for m in models
    }, index=["Best AUC","Mean AUC"]).round(4)
    table_img_path = os.path.join(plots_dir, f"{prefix}_summary_table_colored.png")
    _save_table_as_image(table_df, table_img_path, header_text=header_text)
    logger.log(f"Saved: {table_img_path}")

    top20_csv_path = os.path.join(plots_dir, f"{prefix}_top20_by_model_k.csv")
    top20_df = _save_top20_model_k(perf, top20_csv_path)
    logger.log(f"Saved: {top20_csv_path}")

    report_path = os.path.join(plots_dir, f"{prefix}_report.html")
    imgs = [
        ("AUC vs k (with 95% CI)", line_path),
        ("AUC heatmap", heatmap_path),
        ("AUC distribution per model (boxplot across k)", boxplot_path),
        ("Best/Mean AUC table (image)", table_img_path),
    ]
    tables = [("Top-20 (model, k) by AUC", top20_df.to_html(index=False))]
    _write_report(report_path, logger.html(), imgs, tables, header_text=header_text)
    logger.log(f"Saved: {report_path}")
    logger.log("Done.")

def run(cfg):
    mlplots = cfg.get("mlplots", {})
    results_dir = mlplots.get("results_dir")

    if results_dir is None:
        # 1) intenta la salida por defecto de mlmodels
        out_dir = Path(cfg.get("mlmodels", {}).get("out_dir", "")).as_posix()
        if out_dir:
            results_dir = out_dir
        else:
            # 2) fallback: data/reports/mlmodels bajo paths.reports
            reports_root = Path(cfg.get("paths", {}).get("reports", "data/reports"))
            results_dir = (reports_root / "mlmodels").as_posix()

    print(f"[plots] results_dir = {results_dir}")
    visualize_cv_results(results_dir)
