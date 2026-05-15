"""
Phase 4: Biological Validation
- GSEA enrichment on consensus genes
- Known TME marker overlap
- Spatial visualization of important genes
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
PHASE3_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase3")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TUMOR_MARKERS  = ["KRT8", "KRT18", "KRT19", "EPCAM", "CDH1", "MKI67", "TOP2A"]
STROMA_MARKERS = ["VIM", "COL1A1", "FAP", "ACTA2", "FN1", "COL3A1", "MMP2"]
IMMUNE_MARKERS = ["CD3D", "CD8A", "CD68", "PTPRC", "CD4", "MS4A1", "FOXP3", "TIGIT"]
ALL_KNOWN      = TUMOR_MARKERS + STROMA_MARKERS + IMMUNE_MARKERS

GENE_SETS = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2021",
    "MSigDB_Hallmark_2020",
]

TME_PATHWAYS = [
    "immune response", "extracellular matrix", "angiogenesis",
    "cell proliferation", "inflammatory response", "epithelial",
    "T cell", "cytokine", "tumor", "cancer",
]


def load_consensus_genes():
    path = os.path.join(PHASE3_DIR, "consensus_genes.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run Phase 3 first to generate consensus_genes.csv")
    return pd.read_csv(path)["gene"].tolist()


def run_gsea(gene_list: list) -> pd.DataFrame:
    """Run Enrichr GSEA and return enrichment results."""
    try:
        import gseapy as gp
        print(f"  Running Enrichr on {len(gene_list)} genes ...")
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=GENE_SETS,
            organism="human",
            outdir=os.path.join(OUTPUT_DIR, "enrichr"),
            no_plot=True,
        )
        results = enr.results
        results = results[results["Adjusted P-value"] < 0.05].copy()
        results = results.sort_values("Adjusted P-value")
        print(f"  Significant terms (adj p<0.05): {len(results)}")
        return results
    except ImportError:
        print("  gseapy not installed – skipping GSEA.")
        return pd.DataFrame()
    except Exception as e:
        print(f"  GSEA failed: {e}")
        return pd.DataFrame()


def validate_marker_overlap(consensus_genes: list) -> dict:
    """Measure overlap of consensus genes with known TME markers."""
    consensus_set = set(consensus_genes)
    results = {}
    for name, markers in [("Tumor", TUMOR_MARKERS),
                           ("Stroma", STROMA_MARKERS),
                           ("Immune", IMMUNE_MARKERS),
                           ("All TME", ALL_KNOWN)]:
        overlap = consensus_set & set(markers)
        pct = 100 * len(overlap) / max(len(markers), 1)
        results[name] = {"overlap": len(overlap), "total": len(markers),
                         "pct": pct, "genes": sorted(overlap)}
        print(f"  {name}: {len(overlap)}/{len(markers)} ({pct:.1f}%) "
              f"→ {sorted(overlap)}")
    return results


def plot_marker_overlap(overlap_results: dict) -> None:
    cats = [k for k in overlap_results if k != "All TME"]
    vals = [overlap_results[k]["pct"] for k in cats]
    colors = ["#E05C5C", "#5C9AE0", "#5CE07A"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Overlap (%)")
    ax.set_title("Consensus Gene Overlap with Known TME Markers")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "marker_overlap.png"), dpi=200)
    plt.close()
    print("  Marker overlap plot saved.")


def plot_gsea_results(gsea_df: pd.DataFrame, top_n: int = 15) -> None:
    if gsea_df.empty:
        return
    top = gsea_df.head(top_n).copy()
    top["-log10(adj p)"] = -np.log10(top["Adjusted P-value"].clip(1e-10))
    top["Term"] = top["Term"].str[:60]  # truncate long names

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, x="-log10(adj p)", y="Term", palette="rocket_r", ax=ax)
    ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
    ax.set_title("Top GSEA Enrichment Terms (Consensus Genes)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gsea_top_terms.png"), dpi=200)
    plt.close()
    print("  GSEA plot saved.")


def spatial_gene_importance_plot(adata: sc.AnnData, gene_scores_path: str,
                                 top_n: int = 6) -> None:
    """Spatial plots for top important genes."""
    if not os.path.exists(gene_scores_path):
        return
    scores = pd.read_csv(gene_scores_path, index_col=0, header=None,
                         names=["gene", "importance"]).squeeze()
    top_genes = [g for g in scores.head(top_n).index if g in adata.var_names]

    if not top_genes:
        print("  No top genes found in adata.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, gene in enumerate(top_genes[:6]):
        sc.pl.spatial(adata, color=gene, ax=axes[i], show=False,
                      title=f"{gene} (rank {i+1})", vmin=0)
    plt.suptitle("Spatial Expression of Top Important Genes", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial_top_genes.png"), dpi=200)
    plt.close()
    print("  Spatial gene importance plot saved.")


def run_phase4():
    print("\n" + "="*60)
    print("  PHASE 4: BIOLOGICAL VALIDATION")
    print("="*60 + "\n")

    adata = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))
    consensus_genes = load_consensus_genes()
    print(f"Consensus genes to validate: {len(consensus_genes)}")

    # Marker overlap
    print("\n--- Marker Overlap ---")
    overlap = validate_marker_overlap(consensus_genes)
    plot_marker_overlap(overlap)

    # GSEA
    print("\n--- GSEA Enrichment ---")
    gsea_df = run_gsea(consensus_genes)
    if not gsea_df.empty:
        gsea_df.to_csv(os.path.join(OUTPUT_DIR, "gsea_results.csv"), index=False)
        plot_gsea_results(gsea_df)

        # Check for TME pathways
        tme_hits = gsea_df[gsea_df["Term"].str.lower().apply(
            lambda t: any(p in t for p in TME_PATHWAYS))]
        print(f"\n  TME-relevant pathways enriched: {len(tme_hits)}")
        if not tme_hits.empty:
            print(tme_hits[["Term", "Adjusted P-value"]].head(10).to_string(index=False))

    # Spatial plots
    print("\n--- Spatial Visualization ---")
    gnnex_path = os.path.join(PHASE3_DIR, "gnnexplainer_gene_scores.csv")
    spatial_gene_importance_plot(adata, gnnex_path)

    # Save summary
    summary = {
        "n_consensus_genes": len(consensus_genes),
        "tumor_overlap_pct": overlap.get("Tumor", {}).get("pct", 0),
        "stroma_overlap_pct": overlap.get("Stroma", {}).get("pct", 0),
        "immune_overlap_pct": overlap.get("Immune", {}).get("pct", 0),
        "n_significant_gsea_terms": len(gsea_df),
    }
    pd.Series(summary).to_csv(os.path.join(OUTPUT_DIR, "validation_summary.csv"))
    print(f"\n  Validation summary: {summary}")

    print("\n✅ Phase 4 complete.\n")
    return overlap, gsea_df


if __name__ == "__main__":
    run_phase4()
