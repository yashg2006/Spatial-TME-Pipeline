"""
Phase 5: Cross-Validation & Generalization (GPU-enabled)
- Part A: 5-fold stratified cross-validation
- Part B: Cross-dataset validation on Section 2 (same tissue block)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.sparse import issparse
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
PHASE2_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase5")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED      = 42
N_FOLDS   = 5
N_EPOCHS  = 150
PATIENCE  = 20
LR        = 1e-3
HIDDEN_CH = 128

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── GPU setup ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Model architectures ────────────────────────────────────────────────────────

class GraphSAGE(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_cls, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, hidden_ch)
        self.conv3 = SAGEConv(hidden_ch, n_cls)
        self.bn1   = nn.BatchNorm1d(hidden_ch)
        self.bn2   = nn.BatchNorm1d(hidden_ch)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.log_softmax(self.conv3(x, edge_index), dim=1)


class GCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_cls, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, hidden_ch)
        self.conv3 = GCNConv(hidden_ch, n_cls)
        self.bn1   = nn.BatchNorm1d(hidden_ch)
        self.bn2   = nn.BatchNorm1d(hidden_ch)
        self.drop  = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.log_softmax(self.conv3(x, edge_index, edge_weight), dim=1)


def build_model(arch, in_ch, n_cls):
    if arch == "SAGE":
        return GraphSAGE(in_ch, HIDDEN_CH, n_cls).to(DEVICE)
    elif arch == "GCN":
        return GCN(in_ch, HIDDEN_CH, n_cls).to(DEVICE)
    raise ValueError(f"Unknown architecture: {arch}")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pyg_data():
    """Load long graph, move tensors to DEVICE."""
    print("  Loading Phase 1 data and rebuilding long graph...")
    adata = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))

    adata.obs["tme_label_clean"] = adata.obs["tme_label"].astype(str)
    adata.obs.loc[adata.obs["tme_label_clean"] == "immune",
                  "tme_label_clean"] = "stroma"

    with open(os.path.join(PHASE2_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)
    edge_index, edge_weight = from_scipy_sparse_matrix(
        adata.obsp["spatial_long_connectivities"]
    )
    y_vals = adata.obs["tme_label_clean"].astype(str).values
    y      = torch.tensor(le.transform(y_vals), dtype=torch.long)

    data = Data(
        x=x.to(DEVICE),
        edge_index=edge_index.to(DEVICE),
        edge_weight=edge_weight.float().to(DEVICE),
        y=y.to(DEVICE)
    )
    print(f"  Graph on {DEVICE}: {data.num_nodes} nodes | "
          f"{data.num_edges} edges | classes={le.classes_}")
    return data, le, adata


# ── Training utilities ─────────────────────────────────────────────────────────

def train_epoch(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out      = model(data.x, data.edge_index)
    pred     = out.argmax(dim=1)
    pred_cpu = pred[mask].cpu().numpy()
    true_cpu = data.y[mask].cpu().numpy()
    f1   = f1_score(true_cpu, pred_cpu, average="macro", zero_division=0)
    loss = float(F.nll_loss(out[mask], data.y[mask]))
    return f1, loss, pred_cpu, true_cpu


# ══════════════════════════════════════════════════════════════════════════════
# PART A: 5-FOLD STRATIFIED CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def run_kfold_cv(data, le, arch="SAGE", n_folds=N_FOLDS):
    print(f"\n{'='*60}")
    print(f"  {n_folds}-FOLD CV — {arch} on {DEVICE}")
    print(f"{'='*60}\n")

    n_cls  = len(le.classes_)
    in_ch  = data.x.shape[1]
    labels = data.y.cpu().numpy()   # CPU for sklearn

    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    indices = np.arange(data.num_nodes)

    fold_results       = []
    all_true, all_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"  Fold {fold+1}/{n_folds} — "
              f"train={len(train_idx)} | val={len(val_idx)}")

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        train_mask[train_idx] = True
        val_mask[val_idx]     = True

        torch.manual_seed(SEED + fold)
        model     = build_model(arch, in_ch, n_cls)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                      patience=10, factor=0.5)

        best_val_f1  = 0.0
        best_state   = None
        patience_ctr = 0

        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train_epoch(model, data, train_mask, optimizer)
            val_f1, _, _, _ = evaluate(model, data, val_mask)
            scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                # Save on CPU to avoid VRAM accumulation across folds
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 25 == 0:
                print(f"    Epoch {epoch:3d} | loss={train_loss:.3f} | "
                      f"val F1={val_f1:.4f} | best={best_val_f1:.4f}",
                      flush=True)

            if patience_ctr >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

        # Reload best weights (to GPU)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        final_f1, _, pred, true = evaluate(model, data, val_mask)

        report = classification_report(
            true, pred, target_names=le.classes_,
            output_dict=True, zero_division=0
        )
        row = {"fold": fold+1, "val_f1_macro": final_f1,
               "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum())}
        for cls in le.classes_:
            row[f"f1_{cls}"] = report[cls]["f1-score"]
        fold_results.append(row)
        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())

        cls_scores = ", ".join([f"{c}={report[c]['f1-score']:.3f}" for c in le.classes_])
        print(f"  ✓ Fold {fold+1} F1={final_f1:.4f}  ({cls_scores})")

        # Free GPU memory between folds
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(fold_results)
    mean_f1    = results_df["val_f1_macro"].mean()
    std_f1     = results_df["val_f1_macro"].std()

    print(f"\n  {arch} — F1-Macro: {mean_f1:.4f} ± {std_f1:.4f}")
    for cls in le.classes_:
        col = f"f1_{cls}"
        print(f"  {cls:10s}: {results_df[col].mean():.4f} "
              f"± {results_df[col].std():.4f}")

    return results_df, all_true, all_pred, mean_f1, std_f1


def run_cv_multiple_architectures(data, le):
    all_results  = {}
    summary_rows = []

    for arch in ["SAGE", "GCN"]:
        results_df, all_true, all_pred, mean_f1, std_f1 = run_kfold_cv(
            data, le, arch=arch
        )
        all_results[arch] = {"df": results_df, "true": all_true,
                             "pred": all_pred, "mean": mean_f1, "std": std_f1}
        summary_rows.append({
            "architecture": arch,
            "mean_f1":  round(mean_f1, 4),
            "std_f1":   round(std_f1, 4),
            "ci_lower": round(mean_f1 - 1.96 * std_f1 / np.sqrt(N_FOLDS), 4),
            "ci_upper": round(mean_f1 + 1.96 * std_f1 / np.sqrt(N_FOLDS), 4),
        })
        results_df.to_csv(
            os.path.join(OUTPUT_DIR, f"cv_results_{arch}.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)
    print("\n  Architecture comparison:")
    print(summary_df.to_string(index=False))
    return all_results, summary_df


# ══════════════════════════════════════════════════════════════════════════════
# PART B: SECTION 2 CROSS-DATASET VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_section2(data_dir, adata_ref, le):
    """
    Load Section 2, project onto Section 1's PCA space.
    Valid because both sections are from the same tissue block.
    """
    print(f"  Loading Section 2 from {data_dir}...")
    adata = sc.read_visium(data_dir)
    adata.var_names_make_unique()

    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[adata.obs["total_counts"] > 200]
    adata = adata[:, adata.var["n_cells_by_counts"] > 3]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Align to reference HVGs
    ref_hvgs = adata_ref.var_names[adata_ref.var["highly_variable"]].tolist()
    common   = [g for g in ref_hvgs if g in adata.var_names]
    adata    = adata[:, common]
    print(f"  Common HVGs: {len(common)}/{len(ref_hvgs)} "
          f"({100*len(common)/len(ref_hvgs):.1f}%)")

    if len(common) < len(ref_hvgs) * 0.7:
        raise ValueError(f"Only {len(common)}/{len(ref_hvgs)} HVGs overlap.")

    # Project onto Section 1 PCA space
    sc.pp.scale(adata, max_value=10)
    hvg_idx = [adata_ref.var_names.get_loc(g) for g in common]
    ref_pcs = adata_ref.varm["PCs"][hvg_idx, :]
    X       = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    adata.obsm["X_pca"] = X @ ref_pcs

    # Generate pseudo-labels (merge immune → stroma)
    TUMOR_M  = ["KRT8", "KRT18", "KRT19", "EPCAM", "CDH1", "MUC1", "TFF1"]
    STROMA_M = ["VIM", "COL1A1", "FAP", "ACTA2", "FN1", "COL6A3", "SFRP4"]
    IMMUNE_M = ["CD3D", "CD8A", "CD68", "PTPRC", "CD4"]

    for name, markers in [("tumor", TUMOR_M), ("stroma", STROMA_M),
                           ("immune", IMMUNE_M)]:
        genes = [g for g in markers if g in adata.var_names]
        if genes:
            sc.tl.score_genes(adata, gene_list=genes, score_name=f"{name}_score")

    score_cols = [c for c in ["tumor_score", "stroma_score", "immune_score"]
                  if c in adata.obs.columns]
    if score_cols:
        raw = (adata.obs[score_cols].idxmax(axis=1)
               .str.replace("_score", "", regex=False))
        adata.obs["tme_label_clean"] = raw.replace("immune", "stroma")

    # Build long-range spatial graph
    try:
        import squidpy as sq
        sq.gr.spatial_neighbors(adata, n_neighs=30, key_added="spatial_long")
        print("  ✓ Built spatial_long graph via squidpy")
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix
        coords = adata.obsm["spatial"]
        W = NearestNeighbors(n_neighbors=30).fit(coords).kneighbors_graph(
            coords, mode="connectivity")
        adata.obsp["spatial_long_connectivities"] = csr_matrix(W)
        print("  ✓ Built spatial_long graph via sklearn")

    return adata


@torch.no_grad()
def predict_and_evaluate_section2(model, adata, le):
    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float).to(DEVICE)
    edge_index, _ = from_scipy_sparse_matrix(
        adata.obsp["spatial_long_connectivities"])
    edge_index = edge_index.to(DEVICE)

    model.eval()
    out  = model(x, edge_index)
    pred = out.argmax(dim=1).cpu().numpy()
    pred_labels = le.inverse_transform(pred)
    adata.obs["pred_tme"] = pred_labels

    print("\n  Prediction distribution:")
    print(pd.Series(pred_labels).value_counts().to_string())

    results = {"n_spots": adata.n_obs}
    if "tme_label_clean" not in adata.obs.columns:
        return results

    valid_mask  = adata.obs["tme_label_clean"].isin(le.classes_).values
    true_labels = adata.obs["tme_label_clean"].values[valid_mask]
    pred_subset = pred_labels[valid_mask]

    f1 = f1_score(true_labels, pred_subset, average="macro", zero_division=0)
    print(f"\n  Section 2 F1-macro: {f1:.4f}")
    print(classification_report(true_labels, pred_subset,
                                target_names=le.classes_, zero_division=0))
    results["f1_macro"] = f1
    results["n_valid"]  = int(valid_mask.sum())
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_results(all_results, summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#5C9AE0", "#5CE07A"]

    data_list   = [all_results[a]["df"]["val_f1_macro"].values for a in all_results]
    arch_labels = list(all_results.keys())
    bp = axes[0].boxplot(data_list, labels=arch_labels, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("F1-Macro", fontsize=11)
    axes[0].set_title(f"{N_FOLDS}-Fold CV — F1 Distribution", fontsize=12)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    x     = np.arange(len(summary_df))
    means = summary_df["mean_f1"].values
    stds  = summary_df["std_f1"].values
    bars  = axes[1].bar(x, means, yerr=stds, capsize=8, color=colors,
                        edgecolor="white", linewidth=1.5, alpha=0.85,
                        error_kw={"elinewidth": 2, "ecolor": "black"})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_df["architecture"].values, fontsize=11)
    axes[1].set_ylabel("Mean F1-Macro ± Std", fontsize=11)
    axes[1].set_title("Mean ± Std across folds", fontsize=12)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + s + 0.01,
                     f"{m:.3f}±{s:.3f}",
                     ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Phase 5: Cross-Validation Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cv_results_comparison.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: cv_results_comparison.png")


def plot_fold_f1_curves(all_results):
    for arch, res in all_results.items():
        f1s  = res["df"]["val_f1_macro"].values
        mean = f1s.mean()
        cols = ["#E05C5C" if f < mean else "#5CE07A" for f in f1s]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(res["df"]["fold"].values, f1s, color=cols,
               edgecolor="white", linewidth=1.2)
        ax.axhline(mean, color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean = {mean:.3f}")
        ax.set_xlabel("Fold", fontsize=11)
        ax.set_ylabel("Val F1-Macro", fontsize=11)
        ax.set_title(f"{arch} — Per-fold F1", fontsize=12)
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cv_fold_f1_{arch}.png"),
                    dpi=250, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: cv_fold_f1_{arch}.png")


def plot_confusion_matrix(all_true, all_pred, le, arch="SAGE"):
    cm      = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm], ["Counts", "Normalised"], ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, linewidths=0.5)
        ax.set_title(f"{arch} ({title})", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle(f"Confusion Matrix — {arch} ({N_FOLDS}-fold CV)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{arch}.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: confusion_matrix_{arch}.png")


def plot_spatial_predictions(adata, label="section2"):
    if "spatial" not in adata.obsm or "pred_tme" not in adata.obs.columns:
        return
    coords    = adata.obsm["spatial"]
    color_map = {"tumor": "#E05C5C", "stroma": "#5C9AE0"}

    ncols = 2 if "tme_label_clean" in adata.obs.columns else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    for cls in adata.obs["pred_tme"].unique():
        mask = (adata.obs["pred_tme"] == cls).values
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        c=color_map.get(cls, "gray"), s=6,
                        label=cls, alpha=0.7, linewidths=0)
    axes[0].legend(fontsize=9, markerscale=3)
    axes[0].set_title("SAGE-Long Predictions", fontsize=11)
    axes[0].set_aspect("equal")

    if ncols == 2:
        for cls in adata.obs["tme_label_clean"].unique():
            mask = (adata.obs["tme_label_clean"] == cls).values
            axes[1].scatter(coords[mask, 0], coords[mask, 1],
                            c=color_map.get(cls, "gray"), s=6,
                            label=cls, alpha=0.7, linewidths=0)
        axes[1].legend(fontsize=9, markerscale=3)
        axes[1].set_title("Gene-score Pseudo-labels", fontsize=11)
        axes[1].set_aspect("equal")

    plt.suptitle(f"Section 2 Predictions vs Labels",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"spatial_predictions_{label}.png"),
                dpi=200)
    plt.close()
    print(f"  ✓ Saved: spatial_predictions_{label}.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_phase5():
    print("\n" + "=" * 70)
    print("  PHASE 5: CROSS-VALIDATION & GENERALIZATION")
    print(f"  Device: {DEVICE}")
    print("=" * 70 + "\n")

    data, le, adata_ref = load_pyg_data()

    # ── Part A ───────────────────────────────────────────────────────────────
    all_results, summary_df = run_cv_multiple_architectures(data, le)
    plot_cv_results(all_results, summary_df)
    plot_fold_f1_curves(all_results)
    for arch, res in all_results.items():
        plot_confusion_matrix(res["true"], res["pred"], le, arch=arch)

    print("\n" + "=" * 70)
    print("  PART A SUMMARY")
    print("=" * 70)
    for arch, res in all_results.items():
        print(f"  {arch}: {res['mean']:.4f} ± {res['std']:.4f}")

    # ── Part B ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PART B: SECTION 2 CROSS-DATASET VALIDATION")
    print("=" * 70)

    section2_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "section2")

    if os.path.exists(section2_dir) and any(
        f.endswith(".h5") for f in os.listdir(section2_dir)
    ):
        try:
            adata_s2 = preprocess_section2(section2_dir, adata_ref, le)

            with open(os.path.join(MODEL_DIR, "best_model_metadata.pkl"), "rb") as f:
                meta = pickle.load(f)
            model = build_model("SAGE", meta["in_channels"], meta["num_classes"])
            model.load_state_dict(
                torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                           map_location=DEVICE, weights_only=True))

            s2_results = predict_and_evaluate_section2(model, adata_s2, le)
            plot_spatial_predictions(adata_s2, label="section2")

            if "f1_macro" in s2_results:
                sage_cv = all_results["SAGE"]["mean"]
                gap     = abs(sage_cv - s2_results["f1_macro"])
                verdict = ("✓ Excellent" if gap < 0.05 else
                           "✓ Good"      if gap < 0.10 else
                           "⚠ Large — consider fine-tuning")
                print(f"\n  Section 1 CV F1:    {sage_cv:.4f}")
                print(f"  Section 2 F1:       {s2_results['f1_macro']:.4f}")
                print(f"  Generalization gap: {gap:.4f}  {verdict}")
                s2_results["cv_f1"] = sage_cv
                s2_results["gap"]   = gap

            pd.Series(s2_results).to_csv(
                os.path.join(OUTPUT_DIR, "section2_results.csv"))

        except Exception as e:
            print(f"  ⚠ Section 2 failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"  Section 2 data not found at {section2_dir}/")
        print("\n  Download steps:")
        print("  1. Visit: https://www.10xgenomics.com/resources/datasets/")
        print("            human-breast-cancer-block-a-section-2-1-standard-1-1-0")
        print("  2. Download:")
        print("     - Feature / barcode matrix HDF5 (filtered)  → .h5 file")
        print("     - Spatial imaging data (ZIP)                → extract as spatial/")
        print("  3. Place files in:  data/section2/")
        print("     data/section2/filtered_feature_bc_matrix.h5")
        print("     data/section2/spatial/  (contains positions + images)")
        print("  4. Re-run Phase 5")

    print("\n" + "=" * 70)
    print("  ✅ PHASE 5 COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")

    print("\n📌 KEY NUMBERS FOR YOUR PAPER:")
    for arch, res in all_results.items():
        print(f"  {arch}: F1 = {res['mean']:.4f} ± {res['std']:.4f}")

    return all_results, summary_df


if __name__ == "__main__":
    run_phase5()