"""
Phase 3: Explainability Analysis
- GNNExplainer
- Attention Weight Extraction (from GAT)
- Integrated Gradients (via captum)
- Method comparison + consensus gene set
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
PHASE2_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase3")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_artifacts():
    """Load Phase 2 artefacts."""
    from src.phase2_model import SpatialTME_GAT
    with open(os.path.join(PHASE2_DIR, "pyg_data_local.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(PHASE2_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    adata = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))

    n_cls = len(le.classes_)
    in_ch = data.x.shape[1]
    model = SpatialTME_GAT(in_ch, 128, n_cls)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gat_model.pt"),
                                     map_location="cpu"))
    model.eval()
    return model, data, le, adata


# ── Method 1: GNNExplainer ─────────────────────────────────────────────────────

def run_gnnexplainer(model, data, adata, top_k=20, n_nodes=50):
    """
    Run GNNExplainer on a sample of spots and aggregate gene importance.
    Returns a pd.Series of gene importance scores.
    """
    print("  Running GNNExplainer ...")
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification",
                              task_level="node",
                              return_type="log_probs"),
        )
    except ImportError:
        print("  GNNExplainer import failed – skipping.")
        return pd.Series(dtype=float)

    # Sample spots with valid TME labels
    n_spots = data.num_nodes
    indices = np.random.choice(n_spots, size=min(n_nodes, n_spots), replace=False)

    # HVG gene names
    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()
    n_hvg = sum(hvg_mask)

    # PCA loading matrix  [n_pca x n_hvg]
    pca_loadings = torch.tensor(adata.varm["PCs"][hvg_mask, :].T, dtype=torch.float)

    gene_scores = np.zeros(n_hvg)

    for idx in indices:
        try:
            explanation = explainer(data.x, data.edge_index, index=int(idx))
            feat_mask = explanation.node_mask  # [n_nodes, n_pca]
            # Map PCA feature importance → gene importance
            node_mask_vec = feat_mask[idx].abs()          # [n_pca]
            gene_imp = (pca_loadings.abs() * node_mask_vec.unsqueeze(1)).sum(0)
            gene_scores += gene_imp.numpy()
        except Exception:
            continue

    gene_scores /= max(len(indices), 1)
    result = pd.Series(gene_scores, index=hvg_genes).sort_values(ascending=False)
    top_genes = result.head(top_k)
    print(f"  Top {top_k} GNNExplainer genes: {top_genes.index.tolist()}")
    return result


# ── Method 2: Attention Weights ─────────────────────────────────────────────────

def extract_attention_weights(model, data, adata, top_k=20):
    """Extract per-gene importance from GAT attention weights."""
    print("  Extracting attention weights ...")
    with torch.no_grad():
        _ = model(data.x, data.edge_index, return_attn=True)

    if model.attn1 is None:
        print("  No attention weights stored.")
        return pd.Series(dtype=float)

    # attn1 = (edge_index, alpha)  where alpha: [n_edges, heads]
    edge_idx, alpha = model.attn1
    attn_scores = alpha.abs().mean(dim=1)   # [n_edges]

    # Node-level attention = sum of incoming attention
    n_nodes = data.num_nodes
    node_attn = torch.zeros(n_nodes)
    node_attn.scatter_add_(0, edge_idx[1], attn_scores)

    # Map back to gene space via PCA loadings
    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()
    pca_loadings = torch.tensor(adata.varm["PCs"][hvg_mask, :].T, dtype=torch.float)  # [50, n_hvg]

    # Weight PCA components by node attention aggregated per component
    # Feature-level attention: gradient of output w.r.t. input features (approx)
    # Here: use absolute PCA loadings weighted by overall node attention mean
    node_attn_mean = node_attn.mean()
    gene_scores = (pca_loadings.abs() * node_attn_mean).sum(0).numpy()

    result = pd.Series(gene_scores, index=hvg_genes).sort_values(ascending=False)
    print(f"  Top {top_k} attention-weighted genes: {result.head(top_k).index.tolist()}")
    return result


# ── Method 3: Integrated Gradients ─────────────────────────────────────────────

def run_integrated_gradients(model, data, adata, top_k=20):
    """Run Integrated Gradients via captum."""
    print("  Running Integrated Gradients ...")
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("  captum not installed – skipping IG.")
        return pd.Series(dtype=float)

    def model_wrapper(x):
        return model(x, data.edge_index)

    ig = IntegratedGradients(model_wrapper)
    baseline = torch.zeros_like(data.x)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
    pred_classes = out.argmax(dim=1)

    # Run IG for each unique predicted class
    all_attrs = []
    for cls in pred_classes.unique():
        mask = (pred_classes == cls)
        if mask.sum() == 0:
            continue
        attr = ig.attribute(data.x, baselines=baseline, target=cls.item())
        all_attrs.append(attr[mask].abs().mean(0))  # [n_pca]

    if not all_attrs:
        return pd.Series(dtype=float)

    avg_attr = torch.stack(all_attrs).mean(0)  # [n_pca]

    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()
    pca_loadings = torch.tensor(adata.varm["PCs"][hvg_mask, :].T, dtype=torch.float)
    gene_scores = (pca_loadings.abs() * avg_attr.unsqueeze(1)).sum(0).numpy()

    result = pd.Series(gene_scores, index=hvg_genes).sort_values(ascending=False)
    print(f"  Top {top_k} IG genes: {result.head(top_k).index.tolist()}")
    return result


# ── Consensus & Comparison ─────────────────────────────────────────────────────

def compute_consensus(gnnex_genes, attn_genes, ig_genes, top_k=50):
    """Find genes appearing in 2+ methods (consensus)."""
    s1 = set(gnnex_genes.head(top_k).index)
    s2 = set(attn_genes.head(top_k).index)
    s3 = set(ig_genes.head(top_k).index)
    consensus = (s1 & s2) | (s1 & s3) | (s2 & s3)
    print(f"\n  Consensus genes (≥2 methods): {len(consensus)}")
    print(f"  {sorted(consensus)}")
    return consensus


def plot_method_comparison(gnnex, attn, ig, output_dir, top_n=20):
    """Heatmap comparing top gene scores across methods."""
    all_genes = pd.DataFrame({
        "GNNExplainer": gnnex,
        "Attention":    attn,
        "IntGradients": ig,
    }).fillna(0)

    # Pick top genes by max score across methods
    top_genes = all_genes.max(axis=1).nlargest(top_n).index
    subset = all_genes.loc[top_genes]

    # Normalise each method 0-1
    subset_norm = (subset - subset.min()) / (subset.max() - subset.min() + 1e-9)

    plt.figure(figsize=(10, 8))
    sns.heatmap(subset_norm, cmap="YlOrRd", linewidths=0.5,
                annot=False, cbar_kws={"label": "Normalised Importance"})
    plt.title("Gene Importance Comparison Across Explainability Methods")
    plt.xlabel("Method")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_comparison_heatmap.png"), dpi=200)
    plt.close()
    print("  Comparison heatmap saved.")


def run_phase3():
    print("\n" + "="*60)
    print("  PHASE 3: EXPLAINABILITY ANALYSIS")
    print("="*60 + "\n")

    model, data, le, adata = load_artifacts()

    gnnex_scores = run_gnnexplainer(model, data, adata, top_k=50)
    attn_scores  = extract_attention_weights(model, data, adata, top_k=50)
    ig_scores    = run_integrated_gradients(model, data, adata, top_k=50)

    # Save individual rankings
    for name, ser in [("gnnexplainer", gnnex_scores),
                      ("attention",    attn_scores),
                      ("ig",           ig_scores)]:
        if not ser.empty:
            ser.to_csv(os.path.join(OUTPUT_DIR, f"{name}_gene_scores.csv"),
                       header=["importance"])

    consensus = compute_consensus(gnnex_scores, attn_scores, ig_scores, top_k=50)
    pd.Series(sorted(consensus), name="gene").to_csv(
        os.path.join(OUTPUT_DIR, "consensus_genes.csv"), index=False)

    if not (gnnex_scores.empty or attn_scores.empty or ig_scores.empty):
        plot_method_comparison(gnnex_scores, attn_scores, ig_scores, OUTPUT_DIR)

    print("\n✅ Phase 3 complete.\n")
    return gnnex_scores, attn_scores, ig_scores, consensus


if __name__ == "__main__":
    run_phase3()
