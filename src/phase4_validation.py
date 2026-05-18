"""
Phase 4: Biological Validation & Gene Refinement
- Housekeeping gene filtering
- Differential expression analysis (Wilcoxon)
- Known TME marker overlap with hypergeometric test
- Per-class marker validation
- GSEA enrichment (global + per class)
- Spatial variability filtering (vectorised Moran's I)
- Gene correlation heatmap
- Violin plots
- Comprehensive summary report
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
PHASE3_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase3")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Gene lists ─────────────────────────────────────────────────────────────────

HOUSEKEEPING_GENES = {
    "GAPDH", "ACTB", "ACTG1", "PGK1", "ENO1", "PKM", "LDHA", "LDHB",
    "RPL26", "RPL27", "RPL28", "RPL29", "RPL30", "RPL31", "RPL32", "RPL34",
    "RPL35", "RPL36", "RPL37", "RPL38", "RPL39", "RPL41",
    "RPS2", "RPS3", "RPS4X", "RPS5", "RPS6", "RPS7", "RPS8", "RPS9",
    "RPS10", "RPS11", "RPS12", "RPS13", "RPS14", "RPS15", "RPS16", "RPS17",
    "RPS18", "RPS19", "RPS20", "RPS21", "RPS23", "RPS24", "RPS25",
    "RPS27", "RPS28", "RPS29",
    "TUBA1A", "TUBA1B", "TUBA1C", "TUBB", "TUBB4B",
    "EEF1A1", "EEF1B2", "EEF2", "EIF3E", "EIF4A1",
    "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND1", "MT-ND2", "MT-ND3",
    "MT-ATP6", "MT-CYB",
    "H2AFZ", "H3F3A", "H3F3B",
    "HSP90AA1", "HSP90AB1", "HSPA8", "HSPD1",
}

TUMOR_MARKERS  = ["KRT8", "KRT18", "KRT19", "EPCAM", "CDH1", "MKI67",
                  "TOP2A", "MUC1", "TFF1", "KRT81", "CGA", "KRT7"]
STROMA_MARKERS = ["VIM", "COL1A1", "FAP", "ACTA2", "FN1", "COL3A1",
                  "MMP2", "SFRP4", "COL6A3", "VWF", "ACKR1",
                  "DCN", "SPARC", "PDGFRA", "PDGFRB"]
IMMUNE_MARKERS = ["CD3D", "CD8A", "CD68", "PTPRC", "CD4", "MS4A1",
                  "FOXP3", "TIGIT", "JCHAIN", "MZB1", "IGHG4",
                  "S100A8", "S100A9", "CCL14", "CXCL10", "CCL21",
                  "IL7R", "CD163", "FCGR3A"]
LIVER_MARKERS  = ["ALB", "APOC1", "APOD", "APOE", "CLU", "SERPINA1",
                  "AFP", "TF", "FGB", "CFB", "HBA2", "HBA1", "HBB"]

TME_PATHWAYS = [
    "immune", "extracellular matrix", "angiogenesis", "proliferation",
    "inflammatory", "epithelial", "t cell", "cytokine", "tumor", "cancer",
    "liver", "hepato", "lipid", "complement", "fibroblast",
    "chemokine", "interferon", "apoptosis",
]

GENE_SETS_FALLBACK = [
    ["KEGG_2021_Human", "GO_Biological_Process_2023", "MSigDB_Hallmark_2020"],
    ["KEGG_2019_Human", "GO_Biological_Process_2021", "MSigDB_Hallmark_2020"],
    ["GO_Biological_Process_2021", "MSigDB_Hallmark_2020"],
    ["MSigDB_Hallmark_2020"],
]


# ── Helper ─────────────────────────────────────────────────────────────────────

def get_expr(adata, gene):
    """Return dense 1-D numpy array for a gene."""
    idx = adata.var_names.get_loc(gene)
    x = adata.X[:, idx]
    if issparse(x):
        return x.toarray().flatten()
    if hasattr(x, "toarray"):
        return x.toarray().flatten()
    if hasattr(x, "A"):
        return x.A.flatten()
    return np.array(x).flatten()



def fdr_correction(pvalues):
    """
    FIX: stats.false_discovery_control added in scipy 1.11.
    Falls back to statsmodels or Bonferroni if unavailable.
    """
    try:
        return stats.false_discovery_control(pvalues, method="bh")
    except AttributeError:
        try:
            from statsmodels.stats.multitest import multipletests
            _, padj, _, _ = multipletests(pvalues, method="fdr_bh")
            return padj
        except ImportError:
            print("  ⚠ Using Bonferroni (install statsmodels for BH correction).")
            return np.minimum(pvalues * len(pvalues), 1.0)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_consensus_genes(top_n=100):
    path = os.path.join(PHASE3_DIR, "consensus_borda_ranks.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"consensus_borda_ranks.csv not found in {PHASE3_DIR}")
    df = pd.read_csv(path, index_col=0)
    df.columns = ["mean_rank"]
    genes = df.sort_values("mean_rank").head(top_n).index.tolist()
    print(f"  ✓ Loaded {len(genes)} consensus genes from Borda ranks")
    return genes


def load_per_class_genes(top_n=100):
    per_class = {}
    for cls in ["stroma", "tumor"]:
        path = os.path.join(PHASE3_DIR, f"gnnexplainer_{cls}_gene_scores.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df.columns = ["importance"]
            genes = df.sort_values("importance", ascending=False).head(top_n).index.tolist()
            per_class[cls] = genes
            print(f"  ✓ Loaded {len(genes)} {cls}-specific genes")
        else:
            print(f"  ⚠ {cls} gene scores not found — skipping")
    return per_class


# ── Housekeeping filter ────────────────────────────────────────────────────────

def filter_housekeeping_genes(gene_list, verbose=True):
    """FIX: uses a set for O(1) lookup."""
    filtered = [g for g in gene_list if g not in HOUSEKEEPING_GENES]
    removed  = [g for g in gene_list if g in HOUSEKEEPING_GENES]
    if verbose and removed:
        print(f"\n  Removed {len(removed)} housekeeping genes: {', '.join(removed)}")
        print(f"  Retained: {len(filtered)}/{len(gene_list)}")
    return filtered


# ── Spatial variability ────────────────────────────────────────────────────────

def compute_spatial_variability(adata, genes):
    """
    FIX 1: Vectorised Moran's I using sparse matrix multiply — replaces the
    O(n^2) Python loop from the original (which would take hours).

    FIX 2: Reuses existing spatial graph from adata.obsp instead of
    rebuilding NearestNeighbors from scratch each call.
    """
    # Prefer already-computed spatial graph
    if "spatial_local_connectivities" in adata.obsp:
        W = adata.obsp["spatial_local_connectivities"]
    elif "spatial_connectivities" in adata.obsp:
        W = adata.obsp["spatial_connectivities"]
    else:
        print("  ⚠ No spatial graph in adata.obsp — rebuilding from coordinates...")
        if "spatial" not in adata.obsm:
            print("  ⚠ No spatial coordinates either — skipping spatial filter.")
            return pd.Series(index=genes, data=np.nan)
        from sklearn.neighbors import NearestNeighbors
        coords = adata.obsm["spatial"]
        W = NearestNeighbors(n_neighbors=6).fit(coords).kneighbors_graph(
            coords, mode="connectivity")

    n     = adata.n_obs
    W_sum = float(W.sum())
    valid = [g for g in genes if g in adata.var_names]
    print(f"  Computing Moran's I for {len(valid)} genes (vectorised)...")

    morans_i = {}
    for gene in valid:
        expr  = get_expr(adata, gene)
        xc    = expr - expr.mean()          # mean-centre
        denom = float(xc @ xc)              # x'x
        if denom < 1e-10:
            morans_i[gene] = 0.0
            continue
        Wx  = W @ xc                        # sparse mat-vec: O(edges)
        num = float(xc @ Wx)               # x' W x
        morans_i[gene] = (n / W_sum) * (num / denom)

    result = pd.Series(morans_i).sort_values(ascending=False)
    print(f"  ✓ Moran's I range: {result.min():.3f} – {result.max():.3f}")
    return result


# ── Differential expression ────────────────────────────────────────────────────

def compute_differential_expression(adata, gene_list, label_col="tme_label_clean"):
    """Wilcoxon rank-sum test + BH FDR correction."""
    if label_col not in adata.obs.columns:
        adata.obs["tme_label_clean"] = adata.obs["tme_label"].astype(str)
        adata.obs.loc[adata.obs["tme_label_clean"] == "immune",
                      "tme_label_clean"] = "stroma"
        label_col = "tme_label_clean"

    tumor_mask  = (adata.obs[label_col] == "tumor").values
    stroma_mask = (adata.obs[label_col] == "stroma").values
    valid = [g for g in gene_list if g in adata.var_names]
    print(f"  Computing DE for {len(valid)} genes...")

    rows = []
    for gene in valid:
        expr = get_expr(adata, gene)
        _, pval = stats.ranksums(expr[tumor_mask], expr[stroma_mask])
        tm = expr[tumor_mask].mean()
        sm = expr[stroma_mask].mean()
        rows.append({"gene": gene, "log2fc": np.log2((tm + 1) / (sm + 1)),
                     "pvalue": pval, "tumor_mean": tm, "stroma_mean": sm})

    de_df = pd.DataFrame(rows)
    de_df["padj"] = fdr_correction(de_df["pvalue"].values)   # FIX
    de_df = de_df.sort_values("padj").reset_index(drop=True)
    sig = de_df[de_df["padj"] < 0.05]
    print(f"  ✓ {len(sig)}/{len(valid)} genes significantly DE (padj < 0.05)")
    return de_df


# ── Combined filtering pipeline ────────────────────────────────────────────────

def filter_by_criteria(gene_list, adata, spatial_var_cache=None,
                       min_spatial_var=0.1, min_de_pval=0.05, label=""):
    """
    FIX: Accepts pre-computed spatial_var_cache so Moran's I is only ever
    computed once globally (original called compute_spatial_variability 3×).

    Guards against empty output at each step.
    """
    print(f"\n{'='*60}")
    print(f"  FILTERING: {label or 'gene list'}")
    print(f"{'='*60}")
    print(f"  Starting: {len(gene_list)} genes")

    # Step 1: housekeeping
    filtered = filter_housekeeping_genes(gene_list)

    # Step 2: spatial variability
    if spatial_var_cache is not None:
        spatial_var = spatial_var_cache.reindex(filtered)
    else:
        spatial_var = compute_spatial_variability(adata, filtered)

    if not spatial_var.isna().all():
        keep     = spatial_var[spatial_var >= min_spatial_var].index.tolist()
        filtered = [g for g in filtered if g in keep]
        print(f"  ✓ Spatial (Moran's I ≥ {min_spatial_var}): {len(filtered)} genes")
        if len(filtered) == 0:
            print("  ⚠ All genes removed — relaxing to top-20 by Moran's I")
            filtered = spatial_var.dropna().nlargest(20).index.tolist()
    else:
        print("  ⚠ Spatial filter skipped")

    # Step 3: differential expression
    de_df     = compute_differential_expression(adata, filtered)
    sig_genes = de_df[de_df["padj"] < min_de_pval]["gene"].tolist()

    if len(sig_genes) == 0:
        print("  ⚠ No genes passed DE filter — keeping all spatially variable genes")
        sig_genes = filtered

    filtered = [g for g in filtered if g in sig_genes]
    print(f"  ✓ DE (padj < {min_de_pval}): {len(filtered)} genes retained")
    print(f"\n  Final: {len(filtered)} genes")
    return filtered, de_df, spatial_var


# ── Marker validation ──────────────────────────────────────────────────────────

def validate_marker_overlap(gene_list, label="consensus"):
    gene_set    = set(gene_list)
    marker_sets = {"Tumor": TUMOR_MARKERS, "Stroma": STROMA_MARKERS,
                   "Immune": IMMUNE_MARKERS, "Liver": LIVER_MARKERS}
    results = {}
    print(f"\n  Marker Overlap — {label}")
    print("  " + "-" * 50)
    for name, markers in marker_sets.items():
        found     = sorted(gene_set & set(markers))
        n_overlap = len(found)
        n_total   = len(markers)
        pct       = 100 * n_overlap / max(n_total, 1)
        pval      = stats.hypergeom.sf(n_overlap - 1, 25000, n_total, len(gene_list))
        enriched  = pval < 0.05
        results[name] = {"overlap": n_overlap, "total": n_total,
                         "pct": pct, "genes": found,
                         "pval": pval, "enriched": enriched}
        tag = "✓ ENRICHED" if enriched else ""
        print(f"  {name:8s}: {n_overlap:2d}/{n_total:2d} ({pct:5.1f}%)  "
              f"p={pval:.2e}  {tag}")
        if found:
            print(f"            → {', '.join(found[:6])}")
    return results


def plot_marker_overlap(overlap_results, label="consensus"):
    cats   = list(overlap_results.keys())
    vals   = [overlap_results[k]["pct"] for k in cats]
    pvals  = [overlap_results[k]["pval"] for k in cats]
    colors = ["#E05C5C", "#5C9AE0", "#5CE07A", "#E0B85C"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cats, vals, color=colors, edgecolor="white",
                  linewidth=1.5, alpha=0.85)
    for bar, val, pval in zip(bars, vals, pvals):
        sig = ("***" if pval < 0.001 else "**" if pval < 0.01
               else "*" if pval < 0.05 else "ns")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}%\n{sig}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Overlap (%)", fontsize=11)
    ax.set_title(f"Gene Overlap with Known TME Markers ({label})\n"
                 "* p<0.05  ** p<0.01  *** p<0.001 (hypergeometric)", fontsize=12)
    ax.set_ylim(0, max(vals + [10]) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"marker_overlap_{label}.png")
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(out)}")


def plot_per_class_overlap(per_class_genes):
    if len(per_class_genes) < 2:
        return
    marker_sets = {"Tumor": TUMOR_MARKERS, "Stroma": STROMA_MARKERS,
                   "Immune": IMMUNE_MARKERS, "Liver": LIVER_MARKERS}
    classes = list(per_class_genes.keys())
    x, width = np.arange(len(marker_sets)), 0.35
    colors   = ["#5C9AE0", "#E05C5C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cls in enumerate(classes):
        gs   = set(per_class_genes[cls])
        vals = [100 * len(gs & set(m)) / max(len(m), 1)
                for m in marker_sets.values()]
        ax.bar(x + i * width, vals, width, label=cls.capitalize(),
               color=colors[i], edgecolor="white", alpha=0.85)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(marker_sets.keys(), fontsize=11)
    ax.set_ylabel("Overlap (%)", fontsize=11)
    ax.set_title("Per-class GNNExplainer — Marker Overlap", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "marker_overlap_per_class.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: marker_overlap_per_class.png")


# ── GSEA ───────────────────────────────────────────────────────────────────────

def run_gsea(gene_list, label="consensus"):
    if not gene_list or len(gene_list) < 5:
        print(f"  ⚠ Too few genes ({len(gene_list)}) for GSEA [{label}]")
        return pd.DataFrame()
    try:
        import gseapy as gp
    except ImportError:
        print("  ⚠ gseapy not installed. Run: pip install gseapy")
        return pd.DataFrame()

    outdir = os.path.join(OUTPUT_DIR, f"enrichr_{label}")
    os.makedirs(outdir, exist_ok=True)
    results = pd.DataFrame()

    for gs in GENE_SETS_FALLBACK:
        try:
            print(f"  Running GSEA [{label}] — {len(gene_list)} genes ...")
            enr = gp.enrichr(gene_list=gene_list, gene_sets=gs,
                             organism="human", outdir=outdir,
                             no_plot=True, cutoff=0.05)
            results = enr.results
            if not results.empty:
                print(f"  ✓ Success with: {gs}")
                break
        except Exception as e:
            print(f"  ⚠ Failed ({str(e)[:60]}) — trying next ...")

    if results.empty:
        print(f"  ⚠ GSEA [{label}]: no results.")
        return pd.DataFrame()

    sig = results[results["Adjusted P-value"] < 0.05].copy()
    sig = sig.sort_values("Adjusted P-value")
    tme = sig[sig["Term"].str.lower().apply(
        lambda t: any(kw in t for kw in TME_PATHWAYS))]
    print(f"  ✓ Significant: {len(sig)}  TME-relevant: {len(tme)}")
    return sig


def plot_gsea_results(gsea_df, label="consensus", top_n=20):
    if gsea_df.empty:
        return
    top = gsea_df.head(top_n).copy()
    top["-log10(padj)"] = -np.log10(top["Adjusted P-value"].clip(lower=1e-15))
    top["Term_short"]   = top["Term"].str[:58]

    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.42)))
    palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top)))
    y = np.arange(len(top))
    ax.barh(y, top["-log10(padj)"][::-1], color=palette[::-1],
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(top["Term_short"][::-1], fontsize=9)
    ax.axvline(-np.log10(0.05), color="red", linestyle="--",
               linewidth=1.5, label="adj p=0.05")
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=11)
    ax.set_title(f"Top {len(top)} GSEA Terms — {label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"gsea_top_terms_{label}.png")
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(out)}")


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_gene_expression_violin(adata, genes, label_col="tme_label_clean", top_n=12):
    if label_col not in adata.obs.columns:
        adata.obs[label_col] = adata.obs["tme_label"].astype(str)
        adata.obs.loc[adata.obs[label_col] == "immune", label_col] = "stroma"

    valid = [g for g in genes[:top_n] if g in adata.var_names]
    if not valid:
        return
    ncols = 4
    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for i, gene in enumerate(valid):
        df = pd.DataFrame({"expression": get_expr(adata, gene),
                           "cell_type":  adata.obs[label_col].values})
        sns.violinplot(data=df, x="cell_type", y="expression",
                       ax=axes[i], palette="Set2", inner="box")
        axes[i].set_title(gene, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Expression" if i % ncols == 0 else "")

    for j in range(len(valid), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Gene Expression: Tumor vs Stroma", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gene_expression_violins.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: gene_expression_violins.png")


def plot_gene_correlation_heatmap(adata, genes, top_n=30):
    valid = [g for g in genes[:top_n] if g in adata.var_names]
    if len(valid) < 3:
        print("  ⚠ Too few genes for correlation heatmap.")
        return

    expr_matrix = np.array([get_expr(adata, g) for g in valid])  # [genes × spots]
    corr_matrix = np.corrcoef(expr_matrix)

    # FIX: cluster on correlation distance, not raw expression
    from scipy.spatial.distance import squareform
    dist     = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    link_mat  = linkage(condensed, method="average")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                   gridspec_kw={"width_ratios": [1, 3]})
    dendro = dendrogram(link_mat, orientation="left", ax=ax1,
                        labels=valid, leaf_font_size=8)
    ax1.set_title("Hierarchical clustering", fontsize=10)

    idx     = dendro["leaves"]
    corr_r  = corr_matrix[idx, :][:, idx]
    genes_r = [valid[i] for i in idx]
    sns.heatmap(corr_r, xticklabels=genes_r, yticklabels=genes_r,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "Pearson r"}, ax=ax2, linewidths=0.3)
    ax2.set_title(f"Gene correlation heatmap (top {len(valid)})",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gene_correlation_heatmap.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: gene_correlation_heatmap.png")


def plot_spatial_overview(adata, genes, top_n=9):
    if "spatial" not in adata.obsm:
        print("  ⚠ No spatial coordinates — skipping.")
        return
    valid  = [g for g in genes[:top_n] if g in adata.var_names]
    if not valid:
        return
    coords = adata.obsm["spatial"]
    ncols  = 3
    nrows  = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, gene in enumerate(valid):
        expr = get_expr(adata, gene)
        sc_p = axes[i].scatter(coords[:, 0], coords[:, 1], c=expr,
                               cmap="Reds", s=8, linewidths=0, alpha=0.75)
        plt.colorbar(sc_p, ax=axes[i], shrink=0.7, label="Expression")
        axes[i].set_title(f"{gene} (rank {i+1})", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].set_aspect("equal")

    for j in range(len(valid), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Spatial Expression — Top Refined Genes",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial_refined_genes.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: spatial_refined_genes.png")


# ── Export & report ────────────────────────────────────────────────────────────

def save_refined_gene_lists(refined_global, per_class_refined, de_df):
    def _save(genes, name, de):
        df = pd.DataFrame({"gene": genes, "rank": range(1, len(genes) + 1)})
        if not de.empty:
            df = df.merge(de[["gene", "log2fc", "padj",
                               "tumor_mean", "stroma_mean"]],
                          on="gene", how="left")
        path = os.path.join(OUTPUT_DIR, f"refined_genes_{name}.csv")
        df.to_csv(path, index=False)
        print(f"  ✓ Saved: refined_genes_{name}.csv ({len(genes)} genes)")

    _save(refined_global, "global", de_df)
    for cls, genes in per_class_refined.items():
        _save(genes, cls, de_df)


def generate_summary_report(overlap_global, overlap_per_class, gsea_global, de_df):
    lines = []
    sep = "=" * 70
    lines += [sep, "PHASE 4: BIOLOGICAL VALIDATION SUMMARY", sep, ""]

    lines += ["## MARKER OVERLAP (Refined Consensus)", "-" * 50]
    for cat, info in overlap_global.items():
        tag = "*** ENRICHED ***" if info["enriched"] else ""
        lines.append(f"  {cat:8s}: {info['overlap']:2d}/{info['total']:2d} "
                     f"({info['pct']:.1f}%)  p={info['pval']:.2e}  {tag}")
        if info["genes"]:
            lines.append(f"            {', '.join(info['genes'][:6])}")
    lines.append("")

    if not gsea_global.empty:
        lines += ["## TOP GSEA PATHWAYS", "-" * 50]
        for _, row in gsea_global.head(10).iterrows():
            lines.append(f"  {row['Term'][:65]:<65s}  "
                         f"p={row['Adjusted P-value']:.2e}")
        lines.append("")

    if not de_df.empty:
        sig_up   = de_df[(de_df["padj"] < 0.05) & (de_df["log2fc"] > 1)]
        sig_down = de_df[(de_df["padj"] < 0.05) & (de_df["log2fc"] < -1)]
        lines += ["## DIFFERENTIAL EXPRESSION", "-" * 50,
                  f"  Upregulated in tumor  (log2fc > +1): {len(sig_up)} genes",
                  f"  Upregulated in stroma (log2fc < -1): {len(sig_down)} genes", ""]
        lines.append("  Top tumor-enriched:")
        for _, r in sig_up.head(5).iterrows():
            lines.append(f"    {r['gene']:15s}  log2fc={r['log2fc']:+.2f}  "
                         f"padj={r['padj']:.2e}")
        lines.append("\n  Top stroma-enriched:")
        for _, r in sig_down.head(5).iterrows():
            lines.append(f"    {r['gene']:15s}  log2fc={r['log2fc']:+.2f}  "
                         f"padj={r['padj']:.2e}")

    lines += ["", sep, "END OF REPORT", sep]
    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "validation_report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)
    print("\n  ✓ Saved: validation_report.txt")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_phase4():
    print("\n" + "=" * 70)
    print("  PHASE 4: BIOLOGICAL VALIDATION & GENE REFINEMENT")
    print("=" * 70 + "\n")

    adata               = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))
    consensus_genes_raw = load_consensus_genes(top_n=100)
    per_class_genes_raw = load_per_class_genes(top_n=100)

    # FIX: compute Moran's I ONCE for all genes combined, then cache
    all_genes = list(set(
        consensus_genes_raw
        + [g for gs in per_class_genes_raw.values() for g in gs]
    ))
    all_genes = filter_housekeeping_genes(all_genes, verbose=False)
    print(f"\nPre-computing Moran's I for {len(all_genes)} genes (once only)...")
    spatial_var_cache = compute_spatial_variability(adata, all_genes)

    # ── Filtering ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70 + "\n  GENE REFINEMENT\n" + "=" * 70)

    refined_global, de_df_global, _ = filter_by_criteria(
        consensus_genes_raw, adata,
        spatial_var_cache=spatial_var_cache, label="consensus"
    )
    per_class_refined = {}
    for cls, genes in per_class_genes_raw.items():
        refined, _, _ = filter_by_criteria(
            genes, adata,
            spatial_var_cache=spatial_var_cache, label=cls
        )
        per_class_refined[cls] = refined

    save_refined_gene_lists(refined_global, per_class_refined, de_df_global)

    # ── Marker validation ────────────────────────────────────────────────────
    print("\n" + "=" * 70 + "\n  MARKER VALIDATION\n" + "=" * 70)

    overlap_global = validate_marker_overlap(refined_global, "Refined Consensus")
    plot_marker_overlap(overlap_global, "refined_consensus")

    overlap_per_class = {}
    for cls, genes in per_class_refined.items():
        overlap_per_class[cls] = validate_marker_overlap(genes, f"Refined {cls}")
    if overlap_per_class:
        plot_per_class_overlap(per_class_refined)

    # ── GSEA ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70 + "\n  GSEA ENRICHMENT\n" + "=" * 70)

    gsea_global = run_gsea(refined_global, "refined_consensus")
    if not gsea_global.empty:
        gsea_global.to_csv(
            os.path.join(OUTPUT_DIR, "gsea_refined_consensus.csv"), index=False)
        plot_gsea_results(gsea_global, "refined_consensus", top_n=20)

    for cls, genes in per_class_refined.items():
        gsea_cls = run_gsea(genes, f"refined_{cls}")
        if not gsea_cls.empty:
            gsea_cls.to_csv(
                os.path.join(OUTPUT_DIR, f"gsea_refined_{cls}.csv"), index=False)
            plot_gsea_results(gsea_cls, f"refined_{cls}", top_n=15)

    # ── Visualisations ────────────────────────────────────────────────────────
    print("\n" + "=" * 70 + "\n  VISUALISATIONS\n" + "=" * 70)

    plot_gene_expression_violin(adata, refined_global, top_n=12)
    plot_gene_correlation_heatmap(adata, refined_global, top_n=30)
    plot_spatial_overview(adata, refined_global, top_n=9)

    # ── Summary ───────────────────────────────────────────────────────────────
    generate_summary_report(overlap_global, overlap_per_class,
                            gsea_global, de_df_global)

    print("\n" + "=" * 70)
    print("  ✅ PHASE 4 COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")

    print("\n📌 NEXT STEPS:")
    print("  1. Read validation_report.txt for key findings")
    print("  2. refined_genes_global.csv → final biomarker list")
    print("  3. Check GSEA plots for pathway enrichment story")
    print("  4. Violin plots confirm tumor vs stroma separation")
    print("  5. Proceed to Phase 5: cross-validation")

    return refined_global, overlap_global, gsea_global


if __name__ == "__main__":
    run_phase4()
