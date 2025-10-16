#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import gseapy as gp

from utils_banner import print_banner

# ========================= utilidades =========================

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _split_genes(x):
    if pd.isna(x):
        return []
    return [g.strip() for g in str(x).replace(",", ";").split(";") if g and g.strip() != "."]

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR (q-values)."""
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

def _robust_join_bring_cpg(stats_df: pd.DataFrame, anno_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une tu tabla de estadísticas (con columna 'feature') con la anotación (Anno320K)
    para traer 'CpG' y 'UCSC_RefGene_Name'. Intenta varios emparejamientos robustos.
    """
    s = stats_df.copy()
    if "feature" not in s.columns:
        raise ValueError("La tabla de stats debe tener una columna 'feature' con los IDs de característica.")

    cand = [c for c in anno_df.columns if c != "CpG"]
    vals = set(s["feature"].astype(str).str.strip())
    ov = {c: len(vals & set(anno_df[c].astype(str).str.strip())) for c in cand}
    key = max(ov, key=ov.get) if ov else None

    if not key or ov[key] == 0:
        # prueba quitando ceros a la izquierda
        vals = set(s["feature"].astype(str).str.lstrip("0").str.strip())
        ov = {c: len(vals & set(anno_df[c].astype(str).str.lstrip("0").str.strip())) for c in cand}
        key = max(ov, key=ov.get) if ov else None
        if not key or ov[key] == 0:
            raise ValueError("No se pudo emparejar 'feature' con ninguna columna de Anno para recuperar 'CpG'.")

    s["_key"] = s["feature"].astype(str).str.lstrip("0").str.strip()
    a = anno_df.copy()
    a["_key"] = a[key].astype(str).str.lstrip("0").str.strip()

    m = s.merge(a[["_key", "CpG", "UCSC_RefGene_Name"]], on="_key", how="left").drop(columns=["_key"])
    # filtra CpG válidas
    m = m[m["CpG"].astype(str).str.startswith("cg")].copy()
    m.drop_duplicates(subset=["CpG"], keep="first", inplace=True)
    m["genes_list"] = m["UCSC_RefGene_Name"].apply(_split_genes)
    return m

def _plot_top20_kegg(ora_df: pd.DataFrame, out_png: Path, topN: int = 20, title: str = "KEGG ORA (−log10 p)"):
    """
    Dibuja top-N términos por p (no ajustada). Eje X se auto-ajusta.
    """
    top = ora_df.sort_values("p", ascending=True).head(topN).copy()
    if top.empty:
        return
    plt.figure(figsize=(9, 6))
    vals = -np.log10(top["p"].clip(lower=1e-300))
    plt.barh(top["Term"], vals)
    plt.xlabel("−log10 p (unadjusted)")
    plt.ylabel("KEGG terms")
    plt.title(title)
    plt.gca().invert_yaxis()
    # sin fijar límites → autoscale por defecto
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ========================= paso principal =========================

def run(cfg):
    # ---- Banner ----
    project = str(cfg.get("run", {}).get("project", "default"))
    print_banner(step="functional", project=project)

    # ---- Paths base ----
    paths = cfg.get("paths", {})
    stats_dir = Path(paths.get("reports", "data/reports")).resolve() / "stats" / project
    anno_csv  = Path(paths["annotations"]).resolve()
    out_dir   = Path(paths.get("reports", "data/reports")).resolve() / "functional" / project
    _ensure_dir(out_dir)

    # ---- Parámetros funcionales desde config ----
    func_cfg = cfg.get("functional", {})
    term_query = str(func_cfg.get("term", "Pancreatic cancer"))
    top_n_cpg  = int(func_cfg.get("top_n_cpg", 100))
    # Parámetros adicionales (opcionales)
    rng_seed   = int(func_cfg.get("seed", 7))
    n_perm     = int(func_cfg.get("permutations", 5000))

    # ---- Localiza automáticamente el KS de REC del proyecto ----
    stats_csv = stats_dir / "top_features_ks_rec.csv"
    if not stats_csv.exists():
        raise FileNotFoundError(
            f"No encuentro {stats_csv}. Asegúrate de haber ejecutado 'admeth stats' para el proyecto '{project}'."
        )

    # ---- Carga ----
    # KS-REC con columnas: feature, p_ks, fdr_bh (así lo generó steps_stats)
    ks_rec = pd.read_csv(stats_csv)
    if "feature" not in ks_rec.columns or "p_ks" not in ks_rec.columns:
        raise ValueError(f"{stats_csv} debe contener columnas 'feature' y 'p_ks'.")
    if "fdr_bh" not in ks_rec.columns:
        ks_rec["fdr_bh"] = _bh_fdr(ks_rec["p_ks"].fillna(1.0).to_numpy())

    anno = pd.read_csv(anno_csv)

    # ---- Une para obtener CpG/genes (robusto) ----
    stats_merged = _robust_join_bring_cpg(ks_rec[["feature", "p_ks", "fdr_bh"]], anno)
    stats_merged.to_csv(out_dir / "stats_merged_clean.csv", index=False)
    print(f"[functional] CpGs tras merge/clean: {len(stats_merged):,}")

    # ---- Selección de CpGs TOP_N_CPG por p_ks ----
    sel = stats_merged.sort_values("p_ks", ascending=True).head(top_n_cpg).copy()

    # ---- Lista de genes seleccionados y universo ----
    sel_genes = sorted({g.upper() for lst in sel["genes_list"] for g in lst})
    univ_genes = sorted({g.upper() for lst in stats_merged["genes_list"] for g in lst})
    M = len(univ_genes)
    print(f"[functional] Selected CpGs: {len(sel):,} | Selected genes: {len(sel_genes):,} | Universe genes: {M:,}")

    # ---- ORA con KEGG (hipergeométrico) ----
    lib = gp.get_library(name="KEGG_2021_Human")
    kegg = {t: list({g.upper() for g in gs} & set(univ_genes)) for t, gs in lib.items()}
    
    rows = []
    S = set(sel_genes)
    for term, genes in kegg.items():
        K = len(genes)
        x = len(S & set(genes))
        p = hypergeom.sf(x - 1, M, K, len(S)) if (K > 0 and len(S) > 0) else 1.0
        exp = len(S) * (K / M) if M > 0 else np.nan
        rows.append((term, M, K, len(S), x, exp, p))
    
    ora = pd.DataFrame(rows, columns=["Term", "M_univ", "K_term", "n_sel", "x_overlap", "expected", "p"])
    # (seguimos pudiendo calcular FDR si lo quisieras para uso interno, pero NO lo guardamos)
    # ora["FDR"] = multipletests(ora["p"].values, method="fdr_bh")[1]
    
    ora_out = ora.copy()                 # sin FDR
    ora_path = out_dir / "step2_ORA_KEGG_topCpgs_geneUniverse.csv"
    ora_out.to_csv(ora_path, index=False)
    
    # Filtra término pedido (por si quieres el CSV específico del término)
    term_mask = ora["Term"].str.contains(term_query, case=False, na=False)
    sel_term_df = ora.loc[term_mask].copy()
    sel_term_df.to_csv(out_dir / f"step2_KEGG_{term_query.lower().replace(' ', '_')}_row.csv", index=False)
    
    # Lista de genes solapados con el término (si existe)
    if not sel_term_df.empty:
        pan_term = sel_term_df.iloc[0]["Term"]
        pan_genes = set(kegg.get(pan_term, []))
        overlap_genes = sorted(list(pan_genes & S))
        (out_dir / f"step2_KEGG_{term_query.lower().replace(' ','_')}_overlap_genes.txt").write_text("\n".join(overlap_genes))
    
    # Plot Top-20 con p sin ajustar y autoscale
    _plot_top20_kegg(ora, out_dir / "step2_KEGG_top20.png", topN=20, title="KEGG ORA (−log10 p)")
    print(f"[functional] ORA guardada en: {ora_path}")


    # ---- (Opcional) Permutaciones a nivel CpG (gometh-like) ----
    #   Puedes activar/desactivar vía functional.permutations en config (por defecto 5000).
    rng = np.random.default_rng(rng_seed)
    all_cpg = stats_merged["CpG"].astype(str).unique()
    sel_cpg = sel["CpG"].astype(str).unique()
    k = len(sel_cpg)
    print(f"[functional] Universe CpGs: {len(all_cpg):,} | Selected CpGs: {k:,}")

    # Mapa CpG→genes; KEGG→genes (del universo)
    cpg2genes = {row.CpG: {g.upper() for g in row.genes_list}
                 for row in stats_merged[["CpG", "genes_list"]].itertuples(index=False)}
    universe_genes = {g for s in cpg2genes.values() for g in s}
    kegg2genes = {t: set(g.upper() for g in gs) & universe_genes for t, gs in lib.items()}

    term2cpgs = {}
    for term, genes in kegg2genes.items():
        hits = [cg for cg, gs in cpg2genes.items() if genes & gs]
        term2cpgs[term] = set(hits)

    # Observado
    sel_set = set(sel_cpg)
    obs_rows = []
    for term, cset in term2cpgs.items():
        obs_rows.append((term, len(cset), len(sel_set & cset)))
    obs_df = pd.DataFrame(obs_rows, columns=["Term", "K_cpg", "x_obs"])

    # Permutaciones
    all_cpg_arr = np.array(all_cpg)
    perm_counts = {term: np.zeros(n_perm, dtype=np.int16) for term in term2cpgs}
    for i in range(n_perm):
        samp = set(rng.choice(all_cpg_arr, size=k, replace=False))
        for term, cset in term2cpgs.items():
            perm_counts[term][i] = len(samp & cset)

    rows = []
    for _, r in obs_df.iterrows():
        term, K_cpg, x_obs = r["Term"], int(r["K_cpg"]), int(r["x_obs"])
        arr = perm_counts[term]
        p_emp = (1.0 + (arr >= x_obs).sum()) / (n_perm + 1.0)  # cola derecha
        rows.append((term, K_cpg, x_obs, float(arr.mean()), float(arr.std(ddof=1)), p_emp))

    perm_df = pd.DataFrame(rows, columns=["Term", "K_cpg", "x_obs", "mean_perm", "sd_perm", "p_emp"])
    perm_df["FDR_emp"] = multipletests(perm_df["p_emp"].values, method="fdr_bh")[1]
    perm_df = perm_df.sort_values(["FDR_emp", "p_emp", "x_obs"], ascending=[True, True, False]).reset_index(drop=True)
    perm_path = out_dir / "step3_perm_ORA_KEGG_cpglevel.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"[functional] Permutaciones guardadas en: {perm_path}")

    print(f"✅ [functional] Finalizado. Outputs en: {out_dir}")
