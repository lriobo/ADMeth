#!/usr/bin/env python
# coding: utf-8
import os, base64, json
from pathlib import Path
import pandas as pd
from datetime import datetime
from utils_banner import print_banner

def _b64_img(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            enc = base64.b64encode(f.read()).decode("ascii")
        ext = path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{enc}"
    except Exception:
        return None

def _read_first_csv(glob_paths: list[Path]) -> pd.DataFrame | None:
    for p in glob_paths:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _html_section(title: str, body_html: str) -> str:
    return f"""
    <section style="margin:18px 0;">
      <h2 style="margin:0 0 8px 0;">{title}</h2>
      {body_html}
    </section>
    """

def _html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"
    df_show = df.head(max_rows).copy()
    return df_show.to_html(index=False, escape=False)

def run(cfg):
    # -------- Banner --------
    project = str(cfg.get("run", {}).get("project", "default"))
    print_banner(step="summary", project=project)

    paths = cfg.get("paths", {})
    # Raíces conocidas del proyecto
    mse_root   = Path(paths.get("msemetrics", "data/msemetrics")).resolve() / project
    rec_root   = Path(paths.get("recscores",  "data/recscores")).resolve()  / project
    mlm_root   = Path(paths.get("mlmodels",   "data/reports/mlmodels")).resolve() / project
    stats_root = Path(paths.get("reports",    "data/reports")).resolve() / "stats" / project
    func_root  = Path(paths.get("reports",    "data/reports")).resolve() / "functional" / project
    model_dir  = Path(paths.get("admodel", "models/heavymodelv1")).resolve()

    out_dir = Path(paths.get("reports", "data/reports")).resolve() / "summary" / project
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "ADMeth_summary.html"

    # --------- Cabecera HTML ---------
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_parts = [f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>ADMeth — Summary ({project})</title>
      <style>
        body {{ font-family: Arial, Helvetica, sans-serif; margin: 20px; }}
        h1 {{ margin-bottom: 6px; }}
        h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
        .kbd {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                background:#f5f5f5; border:1px solid #e0e0e0; padding:2px 4px; border-radius:4px; }}
        .row {{ display:flex; gap:14px; flex-wrap:wrap; }}
        .card {{ border:1px solid #e6e6e6; border-radius:8px; padding:12px; min-width:280px; }}
        img {{ max-width:100%; height:auto; border:1px solid #eee; border-radius:6px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border:1px solid #ddd; padding:6px; }}
        th {{ background:#fafafa; }}
        .small {{ color:#666; font-size:12px; }}
      </style>
    </head>
    <body>
      <h1>ADMeth — Project summary</h1>
      <p class="small">Project: <b>{project}</b> • Generated: {now}</p>
    """]

    # --------- Config breve ---------
    html_parts.append(_html_section("Configuration snapshot",
        f"<pre class='kbd'>{json.dumps(cfg.get('run', {}), indent=2, ensure_ascii=False)}</pre>"
    ))

    # --------- Model info ---------
    model_info_txt = model_dir / "model_info.txt"
    if model_info_txt.exists():
        try:
            info = model_info_txt.read_text(encoding="utf-8")
            body = f"<pre class='kbd' style='white-space:pre-wrap'>{info}</pre>"
        except Exception:
            body = "<p><em>No readable model_info.txt</em></p>"
    else:
        body = "<p><em>model_info.txt not found.</em></p>"
    html_parts.append(_html_section("Model info", body))

    # --------- Evaluate (MSE metrics) ---------
    # busca summary por dataset en cases/controls
    mse_rows = []
    for grp in ("cases", "controls"):
        grp_dir = mse_root / grp
        if grp_dir.exists():
            for csv in grp_dir.rglob("*_mse_summary.csv"):
                try:
                    df = pd.read_csv(csv)
                    df.insert(0, "group", grp)
                    df.insert(1, "dataset", csv.parent.name)
                    mse_rows.append(df)
                except Exception:
                    pass
    mse_df = pd.concat(mse_rows, ignore_index=True) if mse_rows else None
    html_parts.append(_html_section("Evaluation (MSE/MAE/RMSE)",
        _html_table(mse_df)))

    # --------- REC scores ---------
    rec_counts = []
    for grp in ("cases", "controls"):
        grp_dir = rec_root / grp
        n = len(list(grp_dir.rglob("*_scores.npy"))) if grp_dir.exists() else 0
        rec_counts.append((grp, n))
    html_parts.append(_html_section("REC scores",
        "<ul>" + "".join([f"<li>{g}: <b>{n}</b> files</li>" for g,n in rec_counts]) + "</ul>"
    ))

    # --------- ML models (plots + combined perf si existe) ---------
    mlm_plots = []
    plots_dir = mlm_root / "plots"
    if plots_dir.exists():
        for name in ["_auc_vs_k_with_ci.png", "_auc_heatmap.png",
                     "_auc_boxplot_per_model.png", "_summary_table_colored.png"]:
            # buscar archivo que termine así
            matches = list(plots_dir.glob(f"*{name}"))
            if matches:
                img64 = _b64_img(matches[0])
                if img64:
                    title = name.strip("_").replace("_"," ").replace(".png","")
                    mlm_plots.append((title, img64))
    body = ""
    if mlm_plots:
        body += "<div class='row'>"
        for title, img in mlm_plots:
            body += f"<div class='card'><h3 style='margin:0 0 8px 0;'>{title}</h3><img src='{img}'></div>"
        body += "</div>"
    else:
        body = "<p><em>No ML model plots found.</em></p>"
    # tabla combinada si existe
    comb_csvs = list((plots_dir).glob("*_combined_performance_long.csv"))
    comb_df = pd.read_csv(comb_csvs[0]) if comb_csvs else None
    body += _html_table(comb_df)
    html_parts.append(_html_section("ML models (CV results)", body))

    # --------- Stats (KS top features + plots) ---------
    tf_rec_csv = stats_root / "top_features_ks_rec.csv"
    tf_raw_csv = stats_root / "top_features_ks_raw.csv"
    tf_rec_df = pd.read_csv(tf_rec_csv) if tf_rec_csv.exists() else None
    tf_raw_df = pd.read_csv(tf_raw_csv) if tf_raw_csv.exists() else None

    stats_plots = []
    sp_dir = stats_root / "plots"
    if sp_dir.exists():
        for name in ["global_density_TRAIN_REC_logY.png",
                     "scatter_medians_TRAIN_REC_symlog.png"]:
            p = sp_dir / name
            if p.exists():
                img = _b64_img(p)
                if img:
                    stats_plots.append((name.replace("_"," ").replace(".png",""), img))

        # primeros 3 boxplots si hay
        feats_dir = sp_dir / "features"
        if feats_dir.exists():
            bxs = sorted(list(feats_dir.glob("feature_*_box*.png")))[:3]
            for p in bxs:
                img = _b64_img(p)
                if img:
                    stats_plots.append((p.name.replace("_"," ").replace(".png",""), img))

    body = "<h3>Top features (REC, KS)</h3>" + _html_table(tf_rec_df, 20)
    body += "<h3>Top features (Raw betas, KS)</h3>" + _html_table(tf_raw_df, 20)
    if stats_plots:
        body += "<div class='row'>"
        for title, img in stats_plots:
            body += f"<div class='card'><h3 style='margin:0 0 8px 0;'>{title}</h3><img src='{img}'></div>"
        body += "</div>"
    html_parts.append(_html_section("Stats (KS & plots)", body))

    # --------- Functional (tablas + top20) ---------
    func_tables = []
    f2 = func_root / "step2_ORA_KEGG_topCpgs_geneUniverse.csv"
    f3 = func_root / "step3_perm_ORA_KEGG_cpglevel.csv"
    if f2.exists():
        func_tables.append(("<b>KEGG ORA (gene universe)</b>", pd.read_csv(f2)))
    if f3.exists():
        func_tables.append(("<b>Permutation ORA (CpG-level)</b>", pd.read_csv(f3)))

    body = ""
    for title, df in func_tables:
        body += f"<h3>{title}</h3>" + _html_table(df, 20)

    # plot top20 si existe
    top20_png = func_root / "step2_KEGG_top20.png"
    if top20_png.exists():
        img = _b64_img(top20_png)
        if img:
            body += f"<div class='card'><h3 style='margin:0 0 8px 0;'>KEGG top-20 (−log10 p)</h3><img src='{img}'></div>"

    html_parts.append(_html_section("Functional analysis", body if body else "<p><em>No functional outputs found.</em></p>"))

    # --------- Cierre ---------
    html_parts.append("</body></html>")
    out_html.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"✅ [summary] Guardado: {out_html}")
