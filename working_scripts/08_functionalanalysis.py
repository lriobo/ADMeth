#!/usr/bin/env python
# coding: utf-8

# In[22]:


# ---- Setup: I/O, params, imports ----
import os, numpy as np, pandas as pd
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from scipy.stats import hypergeom
import gseapy as gp
from tqdm import trange
import matplotlib.pyplot as plt


# In[75]:


# I/O
STATS_CSV = "/home/77462217B/lois/AAImageneAnomalyDetection/plots/Mic3/train_per_feature_tests_with_FDR.csv"     
ANNO_CSV  = "/home/77462217B/lois/AAImageneAnomalyDetection/data/annotations/Anno320K.csv"  
OUTDIR    = Path("functional_v2"); OUTDIR.mkdir(parents=True, exist_ok=True)

# Parameters (edit)
PV_COL       = "p_ks"   # or "p_ks"
FDR_THR_CPG  = 0.1              # step 1: CpG FDR threshold
TOP_N_CPG    = 100             # step 2: top-N CpGs for ORA (set None to use p/FDR threshold instead)
P_THR_CPG    = None              # optional: p-value threshold for step 2 (if TOP_N_CPG=None)
N_PERM       = 5000              # step 3 permutations (increase for precision)
RNG_SEED     = 7

# ---- Load ----
stats = pd.read_csv(STATS_CSV, index_col=0)
anno  = pd.read_csv(ANNO_CSV)


# In[76]:


# ---- Bring CpG & gene annotations (robust) ----
def robust_join_bring_cpg(stats_df: pd.DataFrame, anno_df: pd.DataFrame) -> pd.DataFrame:
    s = stats_df.reset_index()
    s.columns = [("feat" if i==0 else c) for i,c in enumerate(s.columns)]
    cand = [c for c in anno_df.columns if c != "CpG"]
    vals = set(s["feat"].astype(str).str.strip())
    ov   = {c: len(vals & set(anno_df[c].astype(str).str.strip())) for c in cand}
    key  = max(ov, key=ov.get) if ov else None
    if not key or ov[key]==0:
        vals = set(s["feat"].astype(str).str.lstrip("0").str.strip())
        ov   = {c: len(vals & set(anno_df[c].astype(str).str.lstrip("0").str.strip())) for c in cand}
        key  = max(ov, key=ov.get) if ov else None
        if not key or ov[key]==0:
            raise ValueError("Cannot match stats index to any column in Anno320K to get 'CpG'.")
    s["_key"] = s["feat"].astype(str).str.lstrip("0").str.strip()
    a = anno_df.copy(); a["_key"] = a[key].astype(str).str.lstrip("0").str.strip()
    m = s.merge(a[["_key","CpG","UCSC_RefGene_Name"]], on="_key", how="left").drop(columns=["_key"])
    return m

if "CpG" not in stats.columns:
    stats = robust_join_bring_cpg(stats, anno)
elif "UCSC_RefGene_Name" not in stats.columns and "UCSC_RefGene_Name" in anno.columns:
    stats = stats.merge(anno[["CpG","UCSC_RefGene_Name"]], on="CpG", how="left")

# Keep valid CpGs + basic clean
stats = stats[stats["CpG"].astype(str).str.startswith("cg")].copy()
stats = stats.drop_duplicates(subset=["CpG"], keep="first")

# Utility
def split_genes(x):
    if pd.isna(x): return []
    return [g.strip() for g in str(x).replace(",", ";").split(";") if g and g.strip() != "."]

# Precompute gene mapping for later steps
stats["genes_list"] = stats["UCSC_RefGene_Name"].apply(split_genes)
# Save a clean copy
stats.to_csv(OUTDIR/"stats_merged_clean.csv", index=False)
print(f"[INFO] CpGs after merge/clean: {len(stats):,}")


# In[77]:


# ---- Step 1: CpG-level FDR and (optional) |median_diff| filtering ----

# PARAM: absolute median-diff threshold (set 0.0 to disable)
MDIFF_ABS_MIN = 0.00   # e.g., 0.02

# Ensure numeric types
stats[PV_COL] = pd.to_numeric(stats[PV_COL], errors="coerce")
if "median_diff" in stats.columns:
    stats["median_diff"] = pd.to_numeric(stats["median_diff"], errors="coerce")

# CpG-level BH-FDR (compute if missing)
if "FDR_cpg" not in stats.columns:
    stats["FDR_cpg"] = multipletests(stats[PV_COL].values, method="fdr_bh")[1]

# Base mask: FDR
mask_fdr = stats["FDR_cpg"] <= FDR_THR_CPG

# Optional: |median_diff| filter
if MDIFF_ABS_MIN > 0:
    if "median_diff" not in stats.columns:
        raise ValueError(f"median_diff column not found but MDIFF_ABS_MIN={MDIFF_ABS_MIN} > 0. "
                         "Either set MDIFF_ABS_MIN=0 or provide median_diff.")
    stats["abs_median_diff"] = stats["median_diff"].abs()
    mask_mdiff = stats["abs_median_diff"] >= MDIFF_ABS_MIN
else:
    mask_mdiff = np.ones(len(stats), dtype=bool)

# Apply masks
sig_mask = mask_fdr & mask_mdiff

# Tables: (A) only FDR; (B) FDR + mdiff (if threshold > 0)
sig_by_fdr = (stats.loc[mask_fdr, ["CpG", PV_COL, "FDR_cpg", "median_diff", "UCSC_RefGene_Name"]]
                   .sort_values(["FDR_cpg", PV_COL], ascending=[True, True])
                   .reset_index(drop=True))
sig_by_both = (stats.loc[sig_mask, ["CpG", PV_COL, "FDR_cpg", "median_diff", "UCSC_RefGene_Name"]]
                    .sort_values(["FDR_cpg", PV_COL], ascending=[True, True])
                    .reset_index(drop=True))

# Save
sig_by_fdr.to_csv(OUTDIR/"step1_significant_cpgs_byFDR.csv", index=False)
if MDIFF_ABS_MIN > 0:
    # sanitize threshold for filename
    thr_str = str(MDIFF_ABS_MIN).replace(".", "p")
    sig_by_both.to_csv(OUTDIR/f"step1_significant_cpgs_byFDR_mdiff_ge_{thr_str}.csv", index=False)

print(f"[STEP1] Significant CpGs at FDR ≤ {FDR_THR_CPG}: {len(sig_by_fdr):,}")
if MDIFF_ABS_MIN > 0:
    print(f"[STEP1] After |median_diff| ≥ {MDIFF_ABS_MIN}: {len(sig_by_both):,}")

# Display first rows of the active selection for quick check
(sig_by_both if MDIFF_ABS_MIN > 0 else sig_by_fdr).head(10)


# In[78]:


# ---- Step 2: ORA from top CpGs (no 1CpG/gene collapse) ----
# policy: select CpGs by TOP_N_CPG (preferred) or by p/FDR threshold; then build a gene list (unique genes)
# universe = all genes observed in the CpG→gene mapping of 'stats'

# Select CpGs
if TOP_N_CPG is not None:
    sel = stats.sort_values(PV_COL).head(TOP_N_CPG).copy()
else:
    m = np.ones(len(stats), dtype=bool)
    if P_THR_CPG is not None: m &= (stats[PV_COL] <= P_THR_CPG)
    if FDR_THR_CPG is not None: m &= (stats["FDR_cpg"] <= FDR_THR_CPG)
    sel = stats.loc[m].copy()

# Build gene list (unique genes hit by selected CpGs)
sel_genes = sorted({g.upper() for lst in sel["genes_list"] for g in lst})
# Universe genes (all CpGs considered)
univ_genes = sorted({g.upper() for lst in stats["genes_list"] for g in lst})
M = len(univ_genes)
print(f"[STEP2] Selected CpGs: {len(sel):,} | Selected genes: {len(sel_genes):,} | Universe genes: {M:,}")

# Load KEGG and run ORA (hypergeometric) over the gene universe
lib = gp.get_library(name="KEGG_2021_Human")
kegg = {t: list({g.upper() for g in gs} & set(univ_genes)) for t, gs in lib.items()}

rows = []
S = set(sel_genes)
for term, genes in kegg.items():
    K = len(genes)
    x = len(S & set(genes))
    p = hypergeom.sf(x-1, M, K, len(S)) if (K>0 and len(S)>0) else 1.0
    exp = len(S) * (K / M) if M > 0 else np.nan
    rows.append((term, M, K, len(S), x, exp, p))

ora2 = pd.DataFrame(rows, columns=["Term","M_univ","K_term","n_sel","x_overlap","expected","p"])
ora2["FDR"] = multipletests(ora2["p"].values, method="fdr_bh")[1]
ora2 = ora2.sort_values(["FDR","p","x_overlap"], ascending=[True,True,False]).reset_index(drop=True)
ora2.to_csv(OUTDIR/"step2_ORA_KEGG_topCpgs_geneUniverse.csv", index=False)

# Focus on KEGG Pancreatic cancer
kegg_pan = ora2[ora2["Term"].str.contains("Pancreatic cancer", case=False, na=False)]
kegg_pan.to_csv(OUTDIR/"step2_KEGG_pancreatic_cancer_row.csv", index=False)

# Overlap gene list for pancreatic cancer
if not kegg_pan.empty:
    pan_term = kegg_pan.iloc[0]["Term"]
    pan_genes = set(kegg[pan_term])
    overlap_genes = sorted(list(pan_genes & S))
    pd.Series(overlap_genes).to_csv(OUTDIR/"step2_KEGG_pancreatic_cancer_overlap_genes.txt",
                                    index=False, header=False)

# Plot top-20 (no OR displayed)
topN = 20
top_df = ora2.head(topN)
plt.figure(figsize=(9,6))
plt.barh(top_df["Term"], -np.log10(top_df["FDR"].clip(lower=1e-300)))
plt.xlabel("Enrichment score (−log10 adjusted p)")
plt.ylabel("KEGG terms")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTDIR/"step2_KEGG_top20.png", dpi=200); plt.close()

print(f"[STEP2] Saved → {OUTDIR/'step2_ORA_KEGG_topCpgs_geneUniverse.csv'} and plot.")
kegg_pan


# In[79]:


# ---- Step 3: CpG-level permutation ORA (gometh-like) ----
# Idea: keep the selected CpG count k; sample k CpGs from ALL CpGs; count CpG→term hits; empirical p & FDR.
rng = np.random.default_rng(RNG_SEED)

# CpG universe and selected set (from Step 2)
all_cpg = stats["CpG"].astype(str).unique()
sel_cpg = sel["CpG"].astype(str).unique()
k = len(sel_cpg)
print(f"[STEP3] Universe CpGs: {len(all_cpg):,} | Selected CpGs: {k:,}")

# Map CpG→genes, then term→CpGs
cpg2genes = {row.CpG: {g.upper() for g in row.genes_list} for row in stats[["CpG","genes_list"]].itertuples(index=False)}
universe_genes = {g for s in cpg2genes.values() for g in s}
kegg2genes = {t: set(g.upper() for g in gs) & universe_genes for t, gs in lib.items()}

term2cpgs = {}
for term, genes in kegg2genes.items():
    hits = [cg for cg, gs in cpg2genes.items() if genes & gs]
    term2cpgs[term] = set(hits)

# Observed CpG overlaps
sel_set = set(sel_cpg)
obs = []
for term, cset in term2cpgs.items():
    obs.append((term, len(cset), len(sel_set & cset)))
obs_df = pd.DataFrame(obs, columns=["Term","K_cpg","x_obs"])

# Permutations
all_cpg_arr = np.array(all_cpg)
perm_counts = {term: np.zeros(N_PERM, dtype=np.int16) for term in term2cpgs}
for i in trange(N_PERM, desc="Permuting (CpG-level)"):
    samp = set(rng.choice(all_cpg_arr, size=k, replace=False))
    for term, cset in term2cpgs.items():
        perm_counts[term][i] = len(samp & cset)

rows = []
for _, r in obs_df.iterrows():
    term, K_cpg, x_obs = r["Term"], int(r["K_cpg"]), int(r["x_obs"])
    arr = perm_counts[term]
    p_emp = (1.0 + (arr >= x_obs).sum())/(N_PERM + 1.0)  # right-tail
    rows.append((term, K_cpg, x_obs, float(arr.mean()), float(arr.std(ddof=1)), p_emp))

perm_df = pd.DataFrame(rows, columns=["Term","K_cpg","x_obs","mean_perm","sd_perm","p_emp"])
perm_df["FDR_emp"] = multipletests(perm_df["p_emp"].values, method="fdr_bh")[1]
perm_df = perm_df.sort_values(["FDR_emp","p_emp","x_obs"], ascending=[True,True,False]).reset_index(drop=True)
perm_df.to_csv(OUTDIR/"step3_perm_ORA_KEGG_cpglevel.csv", index=False)

# Pancreatic cancer (empirical)
pan_emp = perm_df[perm_df["Term"].str.contains("Pancreatic cancer", case=False, na=False)]
pan_emp.to_csv(OUTDIR/"step3_KEGG_pancreatic_cancer_row_empirical.csv", index=False)

print(f"[STEP3] Saved → {OUTDIR/'step3_perm_ORA_KEGG_cpglevel.csv'}")
pan_emp.head(3)


# In[80]:


# ---- Step 4: complementary analyses ----

# 4a) GO / Reactome / Hallmark ORA (gene-level; same selection as Step 2)
def run_library_ora(gene_list, universe_genes, library_name, tag):
    libX = gp.get_library(name=library_name)
    sets = {t: list({g.upper() for g in gs} & set(universe_genes)) for t, gs in libX.items()}
    rows = []
    S = set(gene_list)
    M = len(universe_genes)
    for term, genes in sets.items():
        K = len(genes); x = len(S & set(genes))
        p = hypergeom.sf(x-1, M, K, len(S)) if (K>0 and len(S)>0) else 1.0
        exp = len(S) * (K / M) if M>0 else np.nan
        rows.append((term, M, K, len(S), x, exp, p))
    df = pd.DataFrame(rows, columns=["Term","M_univ","K_term","n_sel","x_overlap","expected","p"])
    df["FDR"] = multipletests(df["p"].values, method="fdr_bh")[1]
    df = df.sort_values(["FDR","p","x_overlap"], ascending=[True,True,False]).reset_index(drop=True)
    df.to_csv(OUTDIR/f"step4_ORA_{tag}.csv", index=False)
    return df

univ_genes = sorted({g.upper() for lst in stats["genes_list"] for g in lst})
sel_genes = sorted({g.upper() for lst in sel["genes_list"] for g in lst})

go_bp   = run_library_ora(sel_genes, univ_genes, "GO_Biological_Process_2021", "GO_BP")
reacto  = run_library_ora(sel_genes, univ_genes, "Reactome_2022", "Reactome")
hall    = run_library_ora(sel_genes, univ_genes, "MSigDB_Hallmark_2020", "Hallmark")

# 4b) Gene co-annotation network from enriched KEGG terms (projection)
import networkx as nx

# choose enriched terms (empirical or BH); here take Step 3 empirical FDR_emp ≤ 0.10
enr_terms = set(perm_df.loc[perm_df["FDR_emp"]<=0.10, "Term"].tolist())
if not enr_terms:
    # fallback: top 20 by FDR_emp
    enr_terms = set(perm_df.head(20)["Term"].tolist())

# Build bipartite term↔gene using KEGG sets intersected with universe genes
kegg_sets = {t: set({g.upper() for g in gs}) & set(univ_genes) for t, gs in lib.items()}
B = nx.Graph()
for t in enr_terms:
    B.add_node(t, bipartite="term")
    for g in kegg_sets.get(t, []):
        B.add_node(g, bipartite="gene")
        B.add_edge(t, g)

# Project to gene–gene network (shared enriched terms)
genes_nodes = {n for n,d in B.nodes(data=True) if d.get("bipartite")=="gene"}
G = nx.Graph()
for g in genes_nodes:
    G.add_node(g)
for t in enr_terms:
    genes_t = [n for n in B.neighbors(t) if n in genes_nodes]
    for i in range(len(genes_t)):
        for j in range(i+1, len(genes_t)):
            u,v = genes_t[i], genes_t[j]
            G.add_edge(u,v, weight=G.get_edge_data(u,v,{}).get("weight",0)+1)

# Node metrics
deg = dict(G.degree())
wdeg = dict(G.degree(weight="weight"))
cent = nx.degree_centrality(G)

nodes_df = pd.DataFrame({
    "gene": list(genes_nodes),
    "degree": [deg.get(g,0) for g in genes_nodes],
    "w_degree": [wdeg.get(g,0) for g in genes_nodes],
    "cent": [cent.get(g,0.0) for g in genes_nodes]
}).sort_values(["w_degree","degree"], ascending=False)

# Edge list
edges_df = pd.DataFrame([(u,v,d["weight"]) for u,v,d in G.edges(data=True)],
                        columns=["gene_u","gene_v","weight"]).sort_values("weight", ascending=False)

nodes_df.to_csv(OUTDIR/"step4_gene_coannotation_nodes.csv", index=False)
edges_df.to_csv(OUTDIR/"step4_gene_coannotation_edges.csv", index=False)

print(f"[STEP4] GO/Reactome/Hallmark saved. Gene network nodes: {len(nodes_df)}, edges: {len(edges_df)}")


# In[ ]:




