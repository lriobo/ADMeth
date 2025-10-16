#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python
# coding: utf-8
import os, io, base64, json
from pathlib import Path
import numpy as np
import pandas as pd
from utils_banner import print_banner

# ---------- helpers ----------
def _img_to_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def _maybe(path: Path) -> bool:
    return path.exists() and path.is_file()

def _find_first(*candidates: Path) -> Path | None:
    for c in candidates:
        if _maybe(c):
            return c
    return None

def _section(title: str, html_inner: str) -> str:
    return f"""
    <section style="margin:18px 0;">
      <h2 style="margin:0 0 8px 0;">{title}</h2>
      {html_inner}
    </section>
    """

def _kv_table(d: dict) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            v_str = "<pre style='margin:0;white-space:pre-wrap'>" + \
                    (json.dumps(v, indent=2, ensure_ascii=False)) + "</pre>"
        else:
            v_str = str(v)
        rows.append(f"<tr><td><b>{k}</b></td><td>{v_str}</td></tr>")
    return "<table border='0' cellpadding='6'>" + "".join(rows) + "</table>"

def _link_if_exists(label: str, p: Path) -> str:
    if _maybe(p):
        return f"<a href='{p.as_posix()}' target='_blank'>{label}</a>"
    return f"<span style='opacity:.5'>{label} (no encontrado)</span>"

# ---------- main ----------
def run(cfg):
    project = str(cfg.get("run", {}).get("project", "default"))
    print_banner(step="summary", project=project)

    # Rutas base (todas relativas al proyecto actual)
    paths = cfg.get("paths", {})
    reports_root   = Path(paths.get("reports", "data/reports")).resolve()
    mlreports_root = reports_root / "mlmodels" / project
    stats_root     = reports_root / "stats"    / project
    func_root      = reports_root / "functional" / project
    out_html_dir   = reports_root / "summary" / project
    out_html_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_html_dir / "summary.html"

    # ============= 1) Cabecera + parámetros del config.yaml (CAMBIO 5) =============
    # Selecciono parámetros más útiles, pero dejo un bloque con el resto crudo por si quieres
    run_cfg   = cfg.get("run", {})
    mlcfg     = cfg.get("mlmodels", {})
    functional_cfg = cfg.get("functional", {})
    stats_cfg = cfg.get("stats", {})  # por si lo usas

    # Render compacto de parámetros principales
    params_top = {
        "project": project,
        "select.cases": run_cfg.get("select", {}).get("cases", []),
        "select.controls": run_cfg.get("select", {}).get("controls", []),
        "mlmodels.n_splits": mlcfg.get("n_splits", 5),
        "mlmodels.max_features": mlcfg.get("max_features", 500),
        "mlmodels.n_bootstrap": mlcfg.get("n_bootstrap", 1000),
        "mlmodels.beta_values": mlcfg.get("beta_values", False),
        "functional.kegg_focus_term": functional_cfg.get("kegg_focus_term", None),
        "functional.top_n_cpg": functional_cfg.get("top_n_cpg", None),
    }

    # ============= 2) ML models – mejores 10 (CAMBIO 3) =============
    ml_blocks = []
    # Intentamos ambos modos: recscores y (si existe) betas
    for mode in ("recscores", "betas"):
        mode_dir = mlreports_root / mode
        comb_csv = mode_dir / "plots" / f"{mode}_combined_performance_long.csv"
        if not _maybe(comb_csv):
            # compatibilidad: en tu script guardas el CSV con prefijo del directorio; busca de forma recursiva
            found = list((mode_dir / "plots").glob("*combined_performance_long.csv"))
            if found:
                comb_csv = found[0]
        if _maybe(comb_csv):
            df = pd.read_csv(comb_csv)
            # esperamos columnas: k, model, auc_main, mean_auc_boot, ci95_low, ci95_high, auc_plot
            # si no está auc_plot, usamos mean_auc_boot o auc_main
            score_col = "auc_plot" if "auc_plot" in df.columns else ("mean_auc_boot" if "mean_auc_boot" in df.columns else "auc_main")
            top10 = df.sort_values(score_col, ascending=False).head(10).copy()
            ml_blocks.append(_section(
                f"ML models — Top 10 ({mode})",
                top10.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)
            ))

            # Imágenes estándar si existen
            line_png = _find_first(*(mode_dir / "plots").glob("*auc_vs_k_with_ci.png"))
            heat_png = _find_first(*(mode_dir / "plots").glob("*auc_heatmap.png"))
            box_png  = _find_first(*(mode_dir / "plots").glob("*auc_boxplot_per_model.png"))
            imgs = []
            for title, p in (("AUC vs k (95% CI)", line_png),
                             ("AUC heatmap", heat_png),
                             ("AUC boxplot per model", box_png)):
                if p:
                    imgs.append(f"<div style='margin:8px 0'><div><b>{title}</b></div><img style='max-width:100%' src='{_img_to_data_uri(p)}'></div>")
            if imgs:
                ml_blocks.append(_section(f"ML plots ({mode})", "".join(imgs)))

    if not ml_blocks:
        ml_html = "<p style='opacity:.6'>No se encontraron resultados de ML en este proyecto.</p>"
    else:
        ml_html = "".join(ml_blocks)

    # ============= 3) Stats – incluir TODOS los boxplots (CAMBIO 2) =============
    stats_blocks = []
    # Top features KS (REC y Beta-values)
    tf_rec  = stats_root / "top_features_ks_rec.csv"
    tf_beta = stats_root / "top_features_ks_beta.csv"
    
    tf_sections = []
    
    if _maybe(tf_rec):
        df_tf_rec = pd.read_csv(tf_rec)
        head_rec = df_tf_rec.head(15)
        tf_sections.append(_section(
            "Top features (KS, REC)",
            head_rec.to_html(index=False, float_format=lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x)
            + f"<div style='margin-top:6px'>{_link_if_exists('Descargar CSV completo (REC)', tf_rec)}</div>"
        ))
    
    if _maybe(tf_beta):
        df_tf_beta = pd.read_csv(tf_beta)
        head_beta = df_tf_beta.head(15)
        tf_sections.append(_section(
            "Top features (KS, Beta-values)",
            head_beta.to_html(index=False, float_format=lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x)
            + f"<div style='margin-top:6px'>{_link_if_exists('Descargar CSV completo (Beta-values)', tf_beta)}</div>"
        ))
    
    if tf_sections:
        stats_blocks.append("".join(tf_sections))

    plots_dir = stats_root / "plots"
    if plots_dir.exists():
        # global density
        dens_png = _find_first(*(plots_dir.glob("*global_density_TRAIN_REC_logY.png")))
        scat_png = _find_first(*(plots_dir.glob("*scatter_medians_TRAIN_REC_symlog.png")))
        gl_imgs = []
        for title, p in (("Global density (REC, logY)", dens_png),
                         ("Per-feature medians (REC, symlog)", scat_png)):
            if p:
                gl_imgs.append(f"<div style='margin:8px 0'><div><b>{title}</b></div><img style='max-width:100%' src='{_img_to_data_uri(p)}'></div>")
        if gl_imgs:
            stats_blocks.append(_section("Global stats (REC)", "".join(gl_imgs)))

        # (CAMBIO 2) incluir **todas** las figuras de features/
        feats_dir = plots_dir / "features"
        if feats_dir.exists():
            feat_pngs = sorted(feats_dir.glob("*.png"))
            if feat_pngs:
                # insertamos todas, una debajo de otra
                imgs = []
                for p in feat_pngs:
                    imgs.append(f"<div style='margin:10px 0'><img style='max-width:100%' src='{_img_to_data_uri(p)}'></div>")
                stats_blocks.append(_section(f"Per-feature boxplots (todos: {len(feat_pngs)})", "".join(imgs)))

    stats_html = "".join(stats_blocks) if stats_blocks else "<p style='opacity:.6'>No se encontraron resultados de stats.</p>"

    # ============= 4) Functional — KEGG ORA gene universe ordenado por p asc (CAMBIO 4) =============
    func_blocks = []
    step2_csv = func_root / "step2_ORA_KEGG_topCpgs_geneUniverse.csv"
    if _maybe(step2_csv):
        df2 = pd.read_csv(step2_csv)
        # Orden por p ascendente (aunque exista FDR, seguimos tu indicación)
        if "p" in df2.columns:
            df2 = df2.sort_values("p", ascending=True)
        top20 = df2.head(20).copy()
        func_blocks.append(_section("KEGG ORA (gene universe) – top 20 por p-valor",
            top20.to_html(index=False, float_format=lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x) +
            f"<div style='margin-top:6px'>{_link_if_exists('CSV completo', step2_csv)}</div>"
        ))
        # Si existe el barplot del paso 2, lo añadimos
        bar_png = _find_first(*(func_root.glob("step2_KEGG_top20.png")))
        if bar_png:
            func_blocks.append(_section("KEGG ORA barplot",
                f"<img style='max-width:100%' src='{_img_to_data_uri(bar_png)}'>"
            ))
    else:
        func_blocks.append("<p style='opacity:.6'>No se encontraron resultados del análisis funcional.</p>")

    func_html = "".join(func_blocks)

    # ============= 5) Montaje HTML final (CAMBIO 1: se elimina recscores counts) =============
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ADMeth — Summary ({project})</title>
<style>
 body {{ font-family: Arial, Helvetica, sans-serif; margin: 20px; }}
 h1 {{ margin: 0 0 8px 0; }}
 h2 {{ margin: 0 0 6px 0; }}
 .muted {{ opacity:.6 }}
 .card {{ border:1px solid #e5e5e5; border-radius:10px; padding:14px; margin:14px 0; }}
</style>
</head>
<body>
  <h1>ADMeth — Project summary</h1>
  <div class="card">
    <div><b>Project:</b> {project}</div>
    <div><b>Reports root:</b> {reports_root.as_posix()}</div>
  </div>

  <div class="card">
    <h2 style="margin:0 0 8px 0;">Config parámetros</h2>
    {_kv_table(params_top)}
    <details style="margin-top:8px">
      <summary>Ver config completo</summary>
      <pre style="white-space:pre-wrap; font-size:12px;">{json.dumps(cfg, indent=2, ensure_ascii=False)}</pre>
    </details>
  </div>

  { _section("Machine Learning (CV results)", ml_html) }

  { _section("Statistics (REC)", stats_html) }

  { _section("Functional analysis", func_html) }

  <hr/>
  <p class="muted">Generado automáticamente por ADMeth.</p>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Summary guardado en: {out_html}")
