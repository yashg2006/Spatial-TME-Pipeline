"""
Phase 3: Explainability Analysis
- GNNExplainer (on SAGE-Long, per class)
- Gradient Saliency (architecture-agnostic)
- Integrated Gradients via captum (class-weighted)
- Ranked consensus (Borda count)
- Spatial overlay plots on tissue coordinates
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Import Phase 2 utilities to rebuild the graph and use GraphSAGE
from phase2_model import GraphSAGE, build_pyg_data, prepare_labels

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
PHASE2_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase3")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Artifact loading ───────────────────────────────────────────────────────────

def load_artifacts():
    """
    Load Phase 2 artefacts dynamically using Phase 2 utilities.
    We are explicitly loading GraphSAGE as the chosen best model.
    """
    print("  Loading artifacts from Phase 1 and 2...")
    
    # 1. Load AnnData
    adata = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))
    
    # 2. Reapply label strategy (must match Phase 2)
    adata, label_col = prepare_labels(adata, strategy="merge_immune")
    
    # 3. Rebuild the long-range graph PyG Data object
    data, le = build_pyg_data(adata, graph_key="spatial_long", label_col=label_col)

    n_cls = len(le.classes_)
    in_ch = data.x.shape[1]

    # 4. Load the saved SAGE model
    model = GraphSAGE(in_ch, 128, n_cls)
    # Using best_model.pt assuming SAGE was the top performer in Phase 2
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location="cpu", weights_only=True)
    )
    model.eval()

    print(f"  Loaded SAGE-Long | nodes={data.num_nodes} | edges={data.num_edges} | classes={le.classes_}")
    return model, data, le, adata


def get_pca_loadings(adata):
    """
    Return PCA loading matrix and HVG gene names.

    adata.varm['PCs'] shape: [n_genes x n_PCs]
    After HVG masking:       [n_hvg   x n_PCs]
    After .T:                [n_PCs   x n_hvg]   <-- used for weighted sum over PCA dims
    """
    hvg_mask  = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].tolist()
    # [n_PCs x n_hvg]
    pca_loadings = torch.tensor(adata.varm["PCs"][hvg_mask, :].T, dtype=torch.float)
    return pca_loadings, hvg_genes


def pca_to_gene_scores(pca_importance, pca_loadings):
    """
    Map a [n_PCs] importance vector to [n_hvg] gene scores.

    gene_score[g] = sum_k( |loading[k,g]| * importance[k] )
    This correctly weights each PCA component's contribution to each gene.
    """
    gene_scores = (pca_loadings.abs() * pca_importance.unsqueeze(1)).sum(0)
    return gene_scores.numpy()


# ── Method 1: GNNExplainer ─────────────────────────────────────────────────────

def run_gnnexplainer(model, data, adata, top_k=20, n_nodes=50):
    """
    Run GNNExplainer on a stratified sample of spots (per predicted class)
    and return both global and per-class gene importance series.
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
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="log_probs",
            ),
        )
    except ImportError:
        print("  GNNExplainer import failed – skipping.")
        return pd.Series(dtype=float), {}

    pca_loadings, hvg_genes = get_pca_loadings(adata)
    n_hvg = len(hvg_genes)

    with torch.no_grad():
        pred_classes = model(data.x, data.edge_index).argmax(dim=1).numpy()

    # Stratified sampling
    unique_classes = np.unique(pred_classes)
    n_per_class = max(1, n_nodes // len(unique_classes))
    indices = []
    np.random.seed(SEED)
    for cls in unique_classes:
        cls_idx = np.where(pred_classes == cls)[0]
        chosen  = np.random.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())

    class_gene_scores = {int(cls): np.zeros(n_hvg) for cls in unique_classes}
    class_counts      = {int(cls): 0 for cls in unique_classes}
    global_scores     = np.zeros(n_hvg)

    for idx in indices:
        try:
            explanation   = explainer(data.x, data.edge_index, index=int(idx))
            feat_mask     = explanation.node_mask          
            node_mask_vec = feat_mask[idx].abs()           
            gene_imp      = pca_to_gene_scores(node_mask_vec, pca_loadings)

            cls = int(pred_classes[idx])
            class_gene_scores[cls] += gene_imp
            class_counts[cls]      += 1
            global_scores          += gene_imp
        except Exception as e:
            print(f"    GNNExplainer failed at node {idx}: {e}")
            continue

    # Normalise
    total = sum(class_counts.values())
    global_scores /= max(total, 1)
    for cls in unique_classes:
        if class_counts[int(cls)] > 0:
            class_gene_scores[int(cls)] /= class_counts[int(cls)]

    global_result = pd.Series(global_scores, index=hvg_genes).sort_values(ascending=False)
    per_class_results = {
        cls: pd.Series(scores, index=hvg_genes).sort_values(ascending=False)
        for cls, scores in class_gene_scores.items()
    }

    print(f"  Top {top_k} GNNExplainer genes (global): {global_result.head(top_k).index.tolist()}")
    for cls, ser in per_class_results.items():
        print(f"    Class {cls}: {ser.head(10).index.tolist()}")

    return global_result, per_class_results


# ── Method 2: Gradient Saliency ────────────────────────────────────────────────

def run_gradient_saliency(model, data, adata, top_k=20):
    """
    Gradient-based input saliency — works for ANY GNN architecture.
    """
    print("  Running gradient saliency ...")

    pca_loadings, hvg_genes = get_pca_loadings(adata)

    x = data.x.clone().requires_grad_(True)
    out = model(x, data.edge_index)

    pred_classes = out.argmax(dim=1)
    loss = out[range(len(out)), pred_classes].sum()
    loss.backward()

    saliency = x.grad.abs().mean(dim=0).detach()

    gene_scores = pca_to_gene_scores(saliency, pca_loadings)
    result = pd.Series(gene_scores, index=hvg_genes).sort_values(ascending=False)

    print(f"  Top {top_k} saliency genes: {result.head(top_k).index.tolist()}")
    return result


# ── Method 3: Integrated Gradients ─────────────────────────────────────────────

def run_integrated_gradients(model, data, adata, top_k=20):
    """
    Integrated Gradients via captum.
    """
    print("  Running Integrated Gradients ...")
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("  captum not installed – skipping IG.")
        return pd.Series(dtype=float)

    pca_loadings, hvg_genes = get_pca_loadings(adata)

    def model_wrapper(x):
        return model(x, data.edge_index)

    ig       = IntegratedGradients(model_wrapper)
    baseline = torch.zeros_like(data.x)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
    pred_classes = out.argmax(dim=1)

    all_attrs   = []
    class_sizes = []
    for cls in pred_classes.unique():
        mask = (pred_classes == cls)
        if mask.sum() == 0:
            continue
        attr = ig.attribute(data.x, baselines=baseline, target=cls.item())
        all_attrs.append(attr[mask].abs().mean(0))   
        class_sizes.append(mask.sum().float())

    if not all_attrs:
        return pd.Series(dtype=float)

    weights   = torch.stack(class_sizes)
    attrs_mat = torch.stack(all_attrs)                      
    avg_attr  = (attrs_mat * weights.unsqueeze(1)).sum(0) / weights.sum() 

    gene_scores = pca_to_gene_scores(avg_attr, pca_loadings)
    result = pd.Series(gene_scores, index=hvg_genes).sort_values(ascending=False)

    print(f"  Top {top_k} IG genes: {result.head(top_k).index.tolist()}")
    return result


# ── Consensus via Borda count ──────────────────────────────────────────────────

def compute_consensus_ranked(scores_dict, top_k=50):
    """
    Ranked consensus using Borda count (mean rank across methods).
    """
    valid = {k: v for k, v in scores_dict.items() if not v.empty}
    if len(valid) < 2:
        print("  Not enough methods for consensus.")
        return pd.Series(dtype=float)

    ranks = pd.DataFrame({
        name: ser.rank(ascending=False)   
        for name, ser in valid.items()
    }).fillna(len(list(valid.values())[0]) + 1)   

    mean_rank = ranks.mean(axis=1).sort_values()   

    top_genes = mean_rank.head(top_k)
    print(f"\n  Consensus top-{top_k} genes by Borda count:")
    print(f"  {top_genes.index.tolist()}")

    sets = [set(ser.head(top_k).index) for ser in valid.values()]
    overlap_2plus = sets[0].copy()
    for s in sets[1:]:
        overlap_2plus = overlap_2plus.union(s)
    appearing_2plus = set()
    for i, s1 in enumerate(sets):
        for s2 in sets[i+1:]:
            appearing_2plus |= (s1 & s2)
    print(f"  Genes in ≥2 methods (set): {len(appearing_2plus)}")

    return mean_rank


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_method_comparison(scores_dict, output_dir, top_n=20):
    valid = {k: v for k, v in scores_dict.items() if not v.empty}
    if len(valid) < 2:
        return

    all_genes = pd.DataFrame(valid).fillna(0)
    top_genes = all_genes.max(axis=1).nlargest(top_n).index
    subset    = all_genes.loc[top_genes]
    subset_norm = (subset - subset.min()) / (subset.max() - subset.min() + 1e-9)

    plt.figure(figsize=(10, 8))
    sns.heatmap(subset_norm, cmap="YlOrRd", linewidths=0.5,
                annot=False, cbar_kws={"label": "Normalised importance"})
    plt.title("Gene importance comparison across explainability methods")
    plt.xlabel("Method")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_comparison_heatmap.png"), dpi=200)
    plt.close()
    print("  Method comparison heatmap saved.")


def plot_per_class_gnnexplainer(per_class_results, le, output_dir, top_n=15):
    if not per_class_results:
        return

    n_classes = len(per_class_results)
    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6))
    if n_classes == 1:
        axes = [axes]

    for ax, (cls_idx, scores) in zip(axes, per_class_results.items()):
        cls_name = le.classes_[cls_idx] if cls_idx < len(le.classes_) else f"class_{cls_idx}"
        top = scores.head(top_n)
        ax.barh(top.index[::-1], top.values[::-1], color="steelblue")
        ax.set_title(f"GNNExplainer — {cls_name}")
        ax.set_xlabel("Mean gene importance")
        ax.set_ylabel("Gene")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle("Per-class top genes (GNNExplainer, SAGE-Long)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gnnexplainer_per_class.png"), dpi=200)
    plt.close()
    print("  Per-class GNNExplainer plot saved.")


def plot_spatial_gene_overlay(adata, gene_scores_series, output_dir,
                              method_name="consensus", top_n=4):
    if "spatial" not in adata.obsm:
        print("  No spatial coordinates found in adata.obsm['spatial'] — skipping spatial plots.")
        return

    top_genes = [g for g in gene_scores_series.head(top_n).index if g in adata.var_names]
    if not top_genes:
        print("  Top genes not found in adata.var_names — skipping spatial plots.")
        return

    coords = adata.obsm["spatial"]   
    fig, axes = plt.subplots(1, len(top_genes), figsize=(5 * len(top_genes), 5))
    if len(top_genes) == 1:
        axes = [axes]

    for ax, gene in zip(axes, top_genes):
        if gene not in adata.var_names:
            continue
        expr = adata[:, gene].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().flatten()
        else:
            expr = np.array(expr).flatten()

        sc_plot = ax.scatter(coords[:, 0], coords[:, 1],
                             c=expr, cmap="Reds", s=6, linewidths=0)
        plt.colorbar(sc_plot, ax=ax, shrink=0.7, label="Expression")
        ax.set_title(gene, fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    plt.suptitle(f"Spatial expression — top {method_name} genes", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"spatial_overlay_{method_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Spatial overlay plot saved: {out_path}")


def plot_borda_consensus(mean_rank, output_dir, top_n=20):
    if mean_rank.empty:
        return
    top = mean_rank.head(top_n)
    score = top.max() - top + 1

    plt.figure(figsize=(10, 5))
    plt.bar(score.index, score.values, color="mediumseagreen", edgecolor="white")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("Borda score (higher = more important)")
    plt.title(f"Top {top_n} consensus genes — Borda count across all methods")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consensus_borda_bar.png"), dpi=200)
    plt.close()
    print("  Borda consensus bar chart saved.")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_phase3():
    print("\n" + "=" * 60)
    print("  PHASE 3: EXPLAINABILITY ANALYSIS")
    print("=" * 60 + "\n")

    model, data, le, adata = load_artifacts()

    # ── Method 1: GNNExplainer (global + per class) ──
    gnnex_global, gnnex_per_class = run_gnnexplainer(
        model, data, adata, top_k=50, n_nodes=100
    )

    # ── Method 2: Gradient Saliency (replaces attention — works for SAGE) ──
    saliency_scores = run_gradient_saliency(model, data, adata, top_k=50)

    # ── Method 3: Integrated Gradients (class-frequency weighted) ──
    ig_scores = run_integrated_gradients(model, data, adata, top_k=50)

    # ── Save individual rankings ──
    results_to_save = {
        "gnnexplainer": gnnex_global,
        "gradient_saliency": saliency_scores,
        "ig": ig_scores,
    }
    for name, ser in results_to_save.items():
        if not ser.empty:
            ser.to_csv(
                os.path.join(OUTPUT_DIR, f"{name}_gene_scores.csv"),
                header=["importance"]
            )

    # Save per-class GNNExplainer scores
    for cls_idx, ser in gnnex_per_class.items():
        cls_name = le.classes_[cls_idx] if cls_idx < len(le.classes_) else f"class_{cls_idx}"
        ser.to_csv(
            os.path.join(OUTPUT_DIR, f"gnnexplainer_{cls_name}_gene_scores.csv"),
            header=["importance"]
        )

    # ── Borda count consensus ──
    mean_rank = compute_consensus_ranked(results_to_save, top_k=50)
    if not mean_rank.empty:
        mean_rank.to_csv(
            os.path.join(OUTPUT_DIR, "consensus_borda_ranks.csv"),
            header=["mean_rank"]
        )

    # ── Plots ──
    plot_method_comparison(results_to_save, OUTPUT_DIR, top_n=20)
    plot_per_class_gnnexplainer(gnnex_per_class, le, OUTPUT_DIR, top_n=15)
    plot_borda_consensus(mean_rank, OUTPUT_DIR, top_n=20)

    # Spatial overlays for each method's top genes
    if not mean_rank.empty:
        plot_spatial_gene_overlay(adata, mean_rank, OUTPUT_DIR,
                                  method_name="consensus", top_n=4)
    if not gnnex_global.empty:
        plot_spatial_gene_overlay(adata, gnnex_global, OUTPUT_DIR,
                                  method_name="gnnexplainer", top_n=4)

    print("\n✅ Phase 3 complete.\n")
    print("Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")

    return gnnex_global, gnnex_per_class, saliency_scores, ig_scores, mean_rank


if __name__ == "__main__":
    run_phase3()
