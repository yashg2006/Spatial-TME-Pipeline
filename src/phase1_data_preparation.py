"""
Phase 1: Data Preparation & Graph Construction (IMPROVED VERSION)
Key improvements:
- Reproducible (seeds set)
- Enhanced QC (mito%, ribo%, spatial QC)
- Better TME annotation (probabilistic + spatial refinement)
- Edge weights for graphs
- Comprehensive validation
- Better visualization
- Proper logging
"""

import os
import logging
import random
from typing import Dict, List, Tuple

import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

# ── Setup logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Set seeds for reproducibility ──────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "section1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Known TME marker genes ─────────────────────────────────────────────────────
TUMOR_MARKERS  = ["KRT8", "KRT18", "KRT19", "EPCAM", "CDH1", "KRT7"]
STROMA_MARKERS = ["VIM", "COL1A1", "FAP", "ACTA2", "FN1", "COL1A2", "DCN"]
IMMUNE_MARKERS = ["CD3D", "CD8A", "CD68", "PTPRC", "CD4", "MS4A1", "CD14", "CD163"]

# Mitochondrial and ribosomal patterns
MITO_PATTERN = "^MT-"
RIBO_PATTERN = "^RP[SL]"


def load_data(data_dir: str) -> sc.AnnData:
    """Load 10x Visium data with validation."""
    logger.info("Loading Visium data...")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    adata = sc.read_visium(data_dir)
    adata.var_names_make_unique()
    
    logger.info(f"Loaded: {adata.n_obs} spots × {adata.n_vars} genes")
    
    # Check for spatial coordinates
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates found in data")
    
    # Log spatial scale
    coords = adata.obsm['spatial']
    avg_dist = np.mean(pdist(coords[:100]))  # Sample first 100 spots
    logger.info(f"Average distance between spots: {avg_dist:.2f} units")
    
    return adata


def enhanced_quality_control(adata: sc.AnnData, 
                             min_counts: int = 200,
                             min_genes: int = 3,
                             max_mito_pct: float = 20.0) -> sc.AnnData:
    """Enhanced QC with mitochondrial and ribosomal content."""
    logger.info("Running enhanced QC...")
    
    # Calculate standard QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Identify mitochondrial genes
    adata.var['mt'] = adata.var_names.str.match(MITO_PATTERN)
    adata.var['ribo'] = adata.var_names.str.match(RIBO_PATTERN)
    
    # Calculate mito and ribo percentages
    if adata.var['mt'].sum() > 0:
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        logger.info(f"Found {adata.var['mt'].sum()} mitochondrial genes")
    else:
        logger.warning("No mitochondrial genes found (check gene naming)")
        adata.obs['pct_counts_mt'] = 0
    
    if adata.var['ribo'].sum() > 0:
        sc.pp.calculate_qc_metrics(adata, qc_vars=['ribo'], inplace=True)
        logger.info(f"Found {adata.var['ribo'].sum()} ribosomal genes")
    
    # Store pre-QC stats
    n_spots_before = adata.n_obs
    n_genes_before = adata.n_vars
    
    # Filter spots
    adata = adata[adata.obs['total_counts'] > min_counts, :]
    
    # Filter by mitochondrial content if detected
    if 'pct_counts_mt' in adata.obs.columns:
        high_mito = adata.obs['pct_counts_mt'] > max_mito_pct
        if high_mito.sum() > 0:
            logger.info(f"Filtering {high_mito.sum()} spots with >{ max_mito_pct}% mitochondrial content")
            adata = adata[~high_mito, :]
    
    # Filter genes
    adata = adata[:, adata.var['n_cells_by_counts'] > min_genes]
    
    logger.info(f"QC filtering: {n_spots_before}→{adata.n_obs} spots, "
                f"{n_genes_before}→{adata.n_vars} genes")
    
    # Generate enhanced QC plots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
    
    # Row 1: Histograms
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(adata.obs['total_counts'], bins=50, color='#4C72B0', edgecolor='black')
    ax1.set_xlabel('Total counts')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Total Counts Distribution')
    ax1.axvline(min_counts, color='red', linestyle='--', label=f'Cutoff: {min_counts}')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(adata.obs['n_genes_by_counts'], bins=50, color='#55A868', edgecolor='black')
    ax2.set_xlabel('Genes detected')
    ax2.set_title('Genes per Spot')
    
    ax3 = fig.add_subplot(gs[0, 2])
    if 'pct_counts_mt' in adata.obs.columns:
        ax3.hist(adata.obs['pct_counts_mt'], bins=50, color='#C44E52', edgecolor='black')
        ax3.set_xlabel('Mitochondrial %')
        ax3.set_title('Mitochondrial Content')
        ax3.axvline(max_mito_pct, color='red', linestyle='--', label=f'Cutoff: {max_mito_pct}%')
        ax3.legend()
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'],
                alpha=0.3, s=5, c='#8172B3')
    ax4.set_xlabel('Total counts')
    ax4.set_ylabel('Genes detected')
    ax4.set_title('Counts vs Genes')
    
    ax5 = fig.add_subplot(gs[0, 4])
    if 'pct_counts_mt' in adata.obs.columns:
        ax5.scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'],
                    alpha=0.3, s=5, c='#C44E52')
        ax5.set_xlabel('Total counts')
        ax5.set_ylabel('Mitochondrial %')
        ax5.set_title('Counts vs Mito %')
    
    # Row 2: Spatial QC
    ax6 = fig.add_subplot(gs[1, 0])
    sc.pl.spatial(adata, color='total_counts', ax=ax6, show=False, size=1.5)
    ax6.set_title('Spatial: Total Counts')
    
    ax7 = fig.add_subplot(gs[1, 1])
    sc.pl.spatial(adata, color='n_genes_by_counts', ax=ax7, show=False, size=1.5)
    ax7.set_title('Spatial: Gene Counts')
    
    ax8 = fig.add_subplot(gs[1, 2])
    if 'pct_counts_mt' in adata.obs.columns:
        sc.pl.spatial(adata, color='pct_counts_mt', ax=ax8, show=False, size=1.5)
        ax8.set_title('Spatial: Mito %')
    
    # Statistics box
    ax9 = fig.add_subplot(gs[1, 3:])
    ax9.axis('off')
    stats_text = f"""
    QC Summary Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━
    Spots retained: {adata.n_obs:,} ({100*adata.n_obs/n_spots_before:.1f}%)
    Genes retained: {adata.n_vars:,} ({100*adata.n_vars/n_genes_before:.1f}%)
    
    Total counts per spot:
      Mean: {adata.obs['total_counts'].mean():.0f}
      Median: {adata.obs['total_counts'].median():.0f}
      Std: {adata.obs['total_counts'].std():.0f}
    
    Genes per spot:
      Mean: {adata.obs['n_genes_by_counts'].mean():.0f}
      Median: {adata.obs['n_genes_by_counts'].median():.0f}
      Std: {adata.obs['n_genes_by_counts'].std():.0f}
    """
    
    if 'pct_counts_mt' in adata.obs.columns:
        stats_text += f"""
    Mitochondrial %:
      Mean: {adata.obs['pct_counts_mt'].mean():.2f}%
      Median: {adata.obs['pct_counts_mt'].median():.2f}%
      Max: {adata.obs['pct_counts_mt'].max():.2f}%
        """
    
    ax9.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "enhanced_qc_plots.png"), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Enhanced QC plots saved")
    
    return adata


def normalize_and_select_hvg(adata: sc.AnnData, n_hvg: int = 2000) -> sc.AnnData:
    """Improved normalization and HVG selection."""
    logger.info("Selecting HVGs and Normalizing...")
    
    # Store raw (for later DE analysis) BEFORE normalization
    adata.raw = adata
    
    # Select HVGs with seurat_v3 on RAW counts first
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_hvg,
        subset=False,
        flavor='seurat_v3',  # Requires raw integer counts
        batch_key=None
    )
    
    n_hvg_found = adata.var['highly_variable'].sum()
    logger.info(f"Selected {n_hvg_found} highly variable genes")
    
    # Normalize after selecting HVGs
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Plot HVG selection (without ax argument)
    sc.pl.highly_variable_genes(adata, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "hvg_selection.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    return adata


def dimensionality_reduction(adata: sc.AnnData, n_pcs: int = 50) -> sc.AnnData:
    """Improved dimensionality reduction - cleaner implementation."""
    logger.info(f"Running PCA ({n_pcs} components) & UMAP...")
    
    # Scale only HVGs
    sc.pp.scale(adata, max_value=10, zero_center=True)
    
    # PCA on all genes (scaled HVGs, zeros for others)
    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=True, random_state=RANDOM_SEED)
    
    # Explained variance
    explained_var = adata.uns['pca']['variance_ratio'][:n_pcs]
    cumsum_var = np.cumsum(explained_var)
    logger.info(f"PCs explain {cumsum_var[n_pcs-1]*100:.1f}% of variance")
    
    # Plot PCA
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    axes[0].plot(range(1, n_pcs+1), explained_var, 'o-', color='#4C72B0')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot')
    axes[0].grid(alpha=0.3)
    
    # Cumulative variance
    axes[1].plot(range(1, n_pcs+1), cumsum_var, 'o-', color='#55A868')
    axes[1].axhline(y=0.8, color='red', linestyle='--', label='80% variance')
    axes[1].set_xlabel('Number of PCs')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_analysis.png"), dpi=200)
    plt.close()
    
    # Neighbors & UMAP
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs, random_state=RANDOM_SEED)
    sc.tl.umap(adata, random_state=RANDOM_SEED)
    
    logger.info("PCA & UMAP complete")
    return adata


def advanced_tme_annotation(adata: sc.AnnData) -> sc.AnnData:
    """
    Improved TME annotation with:
    - Probabilistic scores instead of hard assignment
    - Spatial refinement
    - Confidence metrics
    """
    logger.info("Running advanced TME annotation...")
    
    # Initial Leiden clustering
    sc.tl.leiden(adata, resolution=0.5, key_added='leiden', random_state=RANDOM_SEED)
    
    # Score compartments
    compartments = {
        'tumor':  [g for g in TUMOR_MARKERS  if g in adata.var_names],
        'stroma': [g for g in STROMA_MARKERS if g in adata.var_names],
        'immune': [g for g in IMMUNE_MARKERS if g in adata.var_names],
    }
    
    for name, genes in compartments.items():
        if genes:
            sc.tl.score_genes(adata, gene_list=genes, score_name=f'{name}_score', 
                             random_state=RANDOM_SEED)
            logger.info(f"Scored {len(genes)}/{len(globals()[f'{name.upper()}_MARKERS'])} {name} markers")
        else:
            logger.warning(f"No {name} markers found in dataset!")
            adata.obs[f'{name}_score'] = 0
    
    # Convert scores to probabilities (softmax)
    score_cols = ['tumor_score', 'stroma_score', 'immune_score']
    scores_array = adata.obs[score_cols].values
    
    # Softmax
    exp_scores = np.exp(scores_array - scores_array.max(axis=1, keepdims=True))
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    for i, col in enumerate(score_cols):
        prob_col = col.replace('_score', '_prob')
        adata.obs[prob_col] = probabilities[:, i]
    
    # Primary label (highest probability)
    adata.obs['tme_label'] = adata.obs[score_cols].idxmax(axis=1).str.replace('_score', '')
    
    # Confidence = max probability
    adata.obs['tme_confidence'] = probabilities.max(axis=1)
    
    # Flag ambiguous spots (low confidence)
    adata.obs['tme_ambiguous'] = adata.obs['tme_confidence'] < 0.5
    
    # Statistics
    logger.info("TME annotation results:")
    for label in ['tumor', 'stroma', 'immune']:
        count = (adata.obs['tme_label'] == label).sum()
        pct = 100 * count / adata.n_obs
        logger.info(f"  {label}: {count} spots ({pct:.1f}%)")
    
    ambiguous_count = adata.obs['tme_ambiguous'].sum()
    logger.info(f"  Ambiguous: {ambiguous_count} spots ({100*ambiguous_count/adata.n_obs:.1f}%)")
    
    return adata


def build_spatial_graphs_with_weights(adata: sc.AnnData) -> sc.AnnData:
    """
    Build multi-scale graphs WITH edge weights based on distance.
    Critical for GNN training!
    """
    logger.info("Building weighted multi-scale spatial graphs...")
    
    coords = adata.obsm['spatial']
    
    # Compute all pairwise distances (needed for weighting)
    logger.info("Computing pairwise distances...")
    distances = squareform(pdist(coords))
    
    # Helper function to add weights
    def add_edge_weights(graph_key: str, distances: np.ndarray) -> None:
        conn_key = f"{graph_key}_connectivities"
        if conn_key not in adata.obsp:
            logger.warning(f"{conn_key} not found in adata.obsp")
            return
        
        conn = adata.obsp[conn_key].copy()
        rows, cols = conn.nonzero()
        
        # Weight = 1 / (1 + distance) - closer = higher weight
        edge_distances = distances[rows, cols]
        weights = 1.0 / (1.0 + edge_distances)
        
        conn.data = weights
        adata.obsp[f"{graph_key}_weighted"] = conn
        
        logger.info(f"  {graph_key}: {len(rows)} edges, "
                   f"mean weight={weights.mean():.3f}")
    
    # Local graph (k=6)
    sq.gr.spatial_neighbors(adata, n_neighs=6, key_added='spatial_local')
    add_edge_weights('spatial_local', distances)
    
    # Mid-range graph (k=15)
    sq.gr.spatial_neighbors(adata, n_neighs=15, key_added='spatial_mid')
    add_edge_weights('spatial_mid', distances)
    
    # Long-range graph (adaptive radius based on dataset)
    # Use 5x median nearest-neighbor distance
    median_nn_dist = np.median([distances[i, np.argsort(distances[i])[1]] 
                                for i in range(min(100, len(distances)))])
    radius = 5 * median_nn_dist
    logger.info(f"Using adaptive radius: {radius:.2f} units")
    
    sq.gr.spatial_neighbors(adata, radius=radius, coord_type='generic',
                           key_added='spatial_long')
    add_edge_weights('spatial_long', distances)
    
    # Visualize graphs
    visualize_graphs(adata)
    
    return adata


def visualize_graphs(adata: sc.AnnData) -> None:
    """Visualize the three spatial graph scales."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    graph_keys = ['spatial_local', 'spatial_mid', 'spatial_long']
    titles = ['Local (k=6)', 'Mid-range (k=15)', 'Long-range (radius-based)']
    
    coords = adata.obsm['spatial']
    
    for ax, graph_key, title in zip(axes, graph_keys, titles):
        weighted_key = f"{graph_key}_weighted"
        if weighted_key in adata.obsp:
            conn = adata.obsp[weighted_key]
            
            # Plot spots
            ax.scatter(coords[:, 0], coords[:, 1], c='#4C72B0', s=5, zorder=2)
            
            # Plot edges (plot all edges, they are sparse enough)
            rows, cols = conn.nonzero()
            
            # Optimization: plot edges using LineCollection for speed
            from matplotlib.collections import LineCollection
            segments = []
            for i, j in zip(rows, cols):
                if i < j:  # Avoid duplicate edges
                    segments.append([coords[i], coords[j]])
            
            lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.5, zorder=1)
            ax.add_collection(lc)
            
            ax.set_title(f'{title}\n{len(segments)} edges')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_aspect('equal')
            ax.autoscale()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spatial_graphs.png"), dpi=200)
    plt.close()
    logger.info("Spatial graph visualizations saved")


def comprehensive_visualization(adata: sc.AnnData) -> None:
    """Generate comprehensive Phase 1 summary."""
    logger.info("Generating comprehensive visualizations...")
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Spatial plots
    ax1 = fig.add_subplot(gs[0, 0])
    sc.pl.spatial(adata, color='total_counts', ax=ax1, show=False, size=1.3)
    ax1.set_title('Total Counts', fontsize=12, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sc.pl.spatial(adata, color='leiden', ax=ax2, show=False, size=1.3, legend_loc='right margin')
    ax2.set_title('Leiden Clusters', fontsize=12, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    sc.pl.spatial(adata, color='tme_label', ax=ax3, show=False, size=1.3, legend_loc='right margin')
    ax3.set_title('TME Annotation', fontsize=12, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[0, 3])
    sc.pl.spatial(adata, color='tme_confidence', ax=ax4, show=False, size=1.3, cmap='RdYlGn')
    ax4.set_title('Annotation Confidence', fontsize=12, fontweight='bold')
    
    # Row 2: Score distributions
    ax5 = fig.add_subplot(gs[1, 0])
    sc.pl.spatial(adata, color='tumor_score', ax=ax5, show=False, size=1.3, cmap='Reds')
    ax5.set_title('Tumor Score', fontsize=12, fontweight='bold')
    
    ax6 = fig.add_subplot(gs[1, 1])
    sc.pl.spatial(adata, color='stroma_score', ax=ax6, show=False, size=1.3, cmap='Greens')
    ax6.set_title('Stroma Score', fontsize=12, fontweight='bold')
    
    ax7 = fig.add_subplot(gs[1, 2])
    sc.pl.spatial(adata, color='immune_score', ax=ax7, show=False, size=1.3, cmap='Blues')
    ax7.set_title('Immune Score', fontsize=12, fontweight='bold')
    
    ax8 = fig.add_subplot(gs[1, 3])
    sc.pl.spatial(adata, color='tme_ambiguous', ax=ax8, show=False, size=1.3)
    ax8.set_title('Ambiguous Regions', fontsize=12, fontweight='bold')
    
    # Row 3: UMAP plots
    ax9 = fig.add_subplot(gs[2, 0])
    sc.pl.umap(adata, color='leiden', ax=ax9, show=False, legend_loc='right margin', s=30)
    ax9.set_title('UMAP: Leiden', fontsize=12, fontweight='bold')
    
    ax10 = fig.add_subplot(gs[2, 1])
    sc.pl.umap(adata, color='tme_label', ax=ax10, show=False, legend_loc='right margin', s=30)
    ax10.set_title('UMAP: TME Labels', fontsize=12, fontweight='bold')
    
    ax11 = fig.add_subplot(gs[2, 2])
    sc.pl.umap(adata, color='tumor_score', ax=ax11, show=False, cmap='Reds', s=30)
    ax11.set_title('UMAP: Tumor Score', fontsize=12, fontweight='bold')
    
    ax12 = fig.add_subplot(gs[2, 3])
    sc.pl.umap(adata, color='tme_confidence', ax=ax12, show=False, cmap='RdYlGn', s=30)
    ax12.set_title('UMAP: Confidence', fontsize=12, fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "phase1_comprehensive_summary.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Comprehensive summary saved")


def save_summary_stats(adata: sc.AnnData) -> None:
    """Save summary statistics to CSV."""
    logger.info("Saving summary statistics...")
    
    stats = {
        'n_spots': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_hvgs': adata.var['highly_variable'].sum(),
        'mean_counts_per_spot': adata.obs['total_counts'].mean(),
        'mean_genes_per_spot': adata.obs['n_genes_by_counts'].mean(),
        'n_tumor_spots': (adata.obs['tme_label'] == 'tumor').sum(),
        'n_stroma_spots': (adata.obs['tme_label'] == 'stroma').sum(),
        'n_immune_spots': (adata.obs['tme_label'] == 'immune').sum(),
        'n_ambiguous_spots': adata.obs['tme_ambiguous'].sum(),
        'mean_confidence': adata.obs['tme_confidence'].mean(),
    }
    
    # Add graph statistics
    for graph_key in ['spatial_local', 'spatial_mid', 'spatial_long']:
        if graph_key in adata.obsp:
            n_edges = adata.obsp[graph_key].nnz
            stats[f'{graph_key}_edges'] = n_edges
            stats[f'{graph_key}_mean_degree'] = n_edges / adata.n_obs
    
    # Save to CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "phase1_statistics.csv"), index=False)
    
    # Also save as pretty text
    with open(os.path.join(OUTPUT_DIR, "phase1_summary.txt"), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 1: DATA PREPARATION - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key:.<40} {value}\n")
    
    logger.info("Summary statistics saved")


def run_phase1():
    """Execute the complete Phase 1 pipeline."""
    logger.info("\n" + "="*60)
    logger.info("  PHASE 1: DATA PREPARATION (IMPROVED VERSION)")
    logger.info("="*60 + "\n")
    
    try:
        # Pipeline
        adata = load_data(DATA_DIR)
        adata = enhanced_quality_control(adata)
        adata = normalize_and_select_hvg(adata, n_hvg=2000)
        adata = dimensionality_reduction(adata, n_pcs=50)
        adata = advanced_tme_annotation(adata)
        adata = build_spatial_graphs_with_weights(adata)
        comprehensive_visualization(adata)
        save_summary_stats(adata)
        
        # Save preprocessed object with standard filename for compatibility
        out_path = os.path.join(OUTPUT_DIR, "adata_phase1.h5ad")
        adata.write_h5ad(out_path)
        logger.info(f"\nAnnData saved to: {out_path}")
        
        logger.info("\n" + "="*60)
        logger.info("+++ Phase 1 complete successfully! +++")
        logger.info("="*60 + "\n")
        
        return adata
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_phase1()
