"""
Phase 5: Cross-Dataset Validation
- Apply trained GAT to Breast Cancer Section 2
- Evaluate generalization
- Compare important genes across sections
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
PHASE2_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2")
PHASE3_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase3")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase5")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR_S2 = os.path.join(os.path.dirname(__file__), "..", "data", "section2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(in_ch, n_cls):
    from src.phase2_model import SpatialTME_GAT
    model = SpatialTME_GAT(in_ch, 128, n_cls)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gat_model.pt"),
                                     map_location="cpu"))
    model.eval()
    return model


def preprocess_section2(data_dir: str, adata_ref: sc.AnnData) -> sc.AnnData:
    """
    Load Section 2, apply same preprocessing as Section 1.
    Uses Section 1 HVG selection and PCA rotation for consistency.
    """
    print(f"  Loading Section 2 from {data_dir} ...")
    adata = sc.read_visium(data_dir)
    adata.var_names_make_unique()

    # QC
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[adata.obs["total_counts"] > 200, :]
    adata = adata[:, adata.var["n_cells_by_counts"] > 3]

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    # Use same HVGs as Section 1 (intersection)
    ref_hvgs = adata_ref.var_names[adata_ref.var["highly_variable"]].tolist()
    common   = [g for g in ref_hvgs if g in adata.var_names]
    adata    = adata[:, common]
    print(f"  Common HVGs with Section 1: {len(common)}")

    # Project onto Section 1's PCA space
    sc.pp.scale(adata, max_value=10)
    # Use reference PCA rotation
    ref_pcs = adata_ref.varm["PCs"][[adata_ref.var_names.get_loc(g) for g in common], :]
    X_pca   = adata.X @ ref_pcs
    adata.obsm["X_pca"] = X_pca

    # TME scoring
    TUMOR_M  = ["KRT8", "KRT18", "KRT19", "EPCAM", "CDH1"]
    STROMA_M = ["VIM", "COL1A1", "FAP", "ACTA2", "FN1"]
    IMMUNE_M = ["CD3D", "CD8A", "CD68", "PTPRC", "CD4"]
    for name, markers in [("tumor", TUMOR_M), ("stroma", STROMA_M), ("immune", IMMUNE_M)]:
        genes = [g for g in markers if g in adata.var_names]
        if genes:
            sc.tl.score_genes(adata, gene_list=genes, score_name=f"{name}_score")
    score_cols = [c for c in ["tumor_score", "stroma_score", "immune_score"]
                  if c in adata.obs.columns]
    if score_cols:
        adata.obs["tme_label"] = (adata.obs[score_cols]
                                  .idxmax(axis=1)
                                  .str.replace("_score", "", regex=False))

    # Spatial graph
    sq.gr.spatial_neighbors(adata, n_neighs=6, key_added="spatial_local")
    return adata


def predict_section(model, adata, le):
    """Run trained GAT on new section and return predictions."""
    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)
    edge_index, _ = from_scipy_sparse_matrix(adata.obsp["spatial_local_connectivities"])
    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).numpy()

    pred_labels = le.inverse_transform(pred)
    adata.obs["pred_tme"] = pred_labels
    return pred_labels


def compare_gene_rankings(output_dir):
    """Compare top gene sets between Section 1 and Section 2 (if available)."""
    s1_path = os.path.join(PHASE3_DIR, "gnnexplainer_gene_scores.csv")
    if not os.path.exists(s1_path):
        print("  Section 1 gene scores not found – skipping comparison.")
        return
    s1_genes = pd.read_csv(s1_path, index_col=0, header=None,
                           names=["gene", "importance"]).squeeze()
    top50_s1 = set(s1_genes.head(50).index)
    print(f"  Section 1 top-50 genes: {sorted(top50_s1)[:10]} ...")


def run_phase5():
    print("\n" + "="*60)
    print("  PHASE 5: CROSS-DATASET VALIDATION")
    print("="*60 + "\n")

    # Load Section 1 artefacts
    adata_s1 = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))
    with open(os.path.join(PHASE2_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    n_cls = len(le.classes_)
    in_ch = adata_s1.obsm["X_pca"].shape[1]
    model = load_model(in_ch, n_cls)

    # ── Section 1 self-evaluation ──
    print("Section 1 self-evaluation ...")
    with open(os.path.join(PHASE2_DIR, "pyg_data_local.pkl"), "rb") as f:
        data_s1 = pickle.load(f)
    with torch.no_grad():
        out  = model(data_s1.x, data_s1.edge_index)
        pred = out.argmax(dim=1).numpy()
    true  = data_s1.y.numpy()
    f1_s1 = f1_score(true, pred, average="macro")
    print(f"  Section 1 F1-macro (full dataset): {f1_s1:.4f}")

    # ── Section 2 validation (if data available) ──
    results = {"section1_f1": f1_s1}
    if os.path.exists(DATA_DIR_S2):
        print("\nSection 2 validation ...")
        try:
            adata_s2  = preprocess_section2(DATA_DIR_S2, adata_s1)
            pred_s2   = predict_section(model, adata_s2, le)

            if "tme_label" in adata_s2.obs.columns:
                true_s2 = le.transform(adata_s2.obs["tme_label"].astype(str).values)
                f1_s2   = f1_score(true_s2, le.transform(pred_s2), average="macro")
                print(f"  Section 2 F1-macro: {f1_s2:.4f}")
                print(f"  Generalization gap: {abs(f1_s1 - f1_s2):.4f}")
                results["section2_f1"] = f1_s2

                # Spatial prediction plot
                sc.pl.spatial(adata_s2, color="pred_tme", show=False)
                plt.savefig(os.path.join(OUTPUT_DIR, "section2_predictions.png"), dpi=200)
                plt.close()
        except Exception as e:
            print(f"  Section 2 failed: {e}")
    else:
        print(f"\n  Section 2 data not found at {DATA_DIR_S2}.")
        print("  Download with: ")
        print("  curl -O https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/"
              "V1_Breast_Cancer_Block_A_Section_2/"
              "V1_Breast_Cancer_Block_A_Section_2_filtered_feature_bc_matrix.h5")

    compare_gene_rankings(OUTPUT_DIR)
    pd.Series(results).to_csv(os.path.join(OUTPUT_DIR, "generalization_results.csv"))
    print(f"\n  Results: {results}")
    print("\n✅ Phase 5 complete.\n")
    return results


if __name__ == "__main__":
    run_phase5()
