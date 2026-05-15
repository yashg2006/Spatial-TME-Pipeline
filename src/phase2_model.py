"""
Phase 2: GNN Model Development & Training
- Baseline: Random Forest
- GCN, GAT, GraphSAGE
- Training loop with evaluation
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Model Architectures ────────────────────────────────────────────────────────

class GCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_cls):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, hidden_ch)
        self.conv3 = GCNConv(hidden_ch, n_cls)
        self.bn1 = nn.BatchNorm1d(hidden_ch)
        self.bn2 = nn.BatchNorm1d(hidden_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        return F.log_softmax(self.conv3(x, edge_index), dim=1)


class SpatialTME_GAT(nn.Module):
    """GAT with stored attention weights for explainability."""
    def __init__(self, in_ch, hidden_ch, n_cls, heads=4, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_ch, hidden_ch, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_ch*heads, hidden_ch, heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden_ch*heads, n_cls, heads=1, dropout=dropout, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_ch * heads)
        self.bn2 = nn.BatchNorm1d(hidden_ch * heads)
        self.attn1 = None
        self.attn2 = None

    def forward(self, x, edge_index, return_attn=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attn:
            x, self.attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attn:
            x, self.attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index)
        x = F.elu(self.bn2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(self.conv3(x, edge_index), dim=1)


class GraphSAGE(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_cls):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, hidden_ch)
        self.conv3 = SAGEConv(hidden_ch, n_cls)
        self.bn1 = nn.BatchNorm1d(hidden_ch)
        self.bn2 = nn.BatchNorm1d(hidden_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        return F.log_softmax(self.conv3(x, edge_index), dim=1)


# ── Utilities ──────────────────────────────────────────────────────────────────

def build_pyg_data(adata, graph_key="spatial_local", label_col="tme_label"):
    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)
    conn_key = f"{graph_key}_connectivities"
    edge_index, _ = from_scipy_sparse_matrix(adata.obsp[conn_key])
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(adata.obs[label_col].astype(str).values), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.label_names = list(le.classes_)
    return data, le


def train_step(model, data, optimizer, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[mask] == data.y[mask]).float().mean().item()
    return acc, pred[mask].numpy(), data.y[mask].numpy()


def train_model(model, data, n_epochs=200, lr=1e-3, name="model"):
    n = data.num_nodes
    perm = torch.randperm(n)
    split = int(0.8 * n)
    tr_mask = torch.zeros(n, dtype=torch.bool)
    te_mask  = torch.zeros(n, dtype=torch.bool)
    tr_mask[perm[:split]] = True
    te_mask[perm[split:]] = True

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    losses, tr_accs, te_accs = [], [], []
    for ep in range(1, n_epochs + 1):
        loss = train_step(model, data, opt, tr_mask)
        tr_acc, _, _ = evaluate(model, data, tr_mask)
        te_acc, _, _ = evaluate(model, data, te_mask)
        sch.step()
        losses.append(loss); tr_accs.append(tr_acc); te_accs.append(te_acc)
        if ep % 40 == 0:
            print(f"  [{name}] ep {ep:3d} | loss {loss:.4f} | train {tr_acc:.3f} | test {te_acc:.3f}")

    _, pred, true = evaluate(model, data, te_mask)
    f1 = f1_score(true, pred, average="macro")
    print(f"  [{name}] Final F1-macro: {f1:.4f}")
    print(classification_report(true, pred, target_names=data.label_names))
    return {"name": name, "model": model, "losses": losses,
            "te_accs": te_accs, "f1_macro": f1,
            "tr_mask": tr_mask, "te_mask": te_mask}


def run_phase2():
    print("\n" + "="*60)
    print("  PHASE 2: MODEL DEVELOPMENT")
    print("="*60 + "\n")

    adata = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))
    n_cls  = adata.obs["tme_label"].nunique()
    in_ch  = adata.obsm["X_pca"].shape[1]
    hid    = 128
    print(f"Classes: {n_cls} | PCA dims: {in_ch}")

    data_local, le = build_pyg_data(adata, "spatial_local")
    data_mid,   _  = build_pyg_data(adata, "spatial_mid")

    # Random Forest baseline
    X = adata.obsm["X_pca"]
    y_le = LabelEncoder()
    y = y_le.fit_transform(adata.obs["tme_label"].astype(str).values)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_f1s = [f1_score(y[te], RandomForestClassifier(200, n_jobs=-1, random_state=42).fit(X[tr], y[tr]).predict(X[te]), average="macro") for tr, te in skf.split(X, y)]
    print(f"  RF F1-macro: {np.mean(rf_f1s):.4f} ± {np.std(rf_f1s):.4f}")

    # GNN models
    gat  = SpatialTME_GAT(in_ch, hid, n_cls)
    gcn  = GCN(in_ch, hid, n_cls)
    sage = GraphSAGE(in_ch, hid, n_cls)

    gat_r  = train_model(gat,  data_local, n_epochs=200, name="GAT-Local")
    gcn_r  = train_model(gcn,  data_local, n_epochs=200, name="GCN-Local")
    sage_r = train_model(sage, data_mid,   n_epochs=200, name="SAGE-Mid")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in [gat_r, gcn_r, sage_r]:
        axes[0].plot(r["losses"],  label=r["name"])
        axes[1].plot(r["te_accs"], label=r["name"])
    for ax, title in zip(axes, ["Training Loss", "Test Accuracy"]):
        ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=200)
    plt.close()

    # Summary
    rows = [{"Model": "RandomForest", "F1-Macro": f"{np.mean(rf_f1s):.4f}", "Graph": "None"}]
    for r in [gat_r, gcn_r, sage_r]:
        rows.append({"Model": r["name"], "F1-Macro": f"{r['f1_macro']:.4f}", "Graph": r["name"].split("-")[-1]})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # Save artefacts for Phase 3
    torch.save(gat.state_dict(), os.path.join(MODEL_DIR, "gat_model.pt"))
    with open(os.path.join(OUTPUT_DIR, "pyg_data_local.pkl"), "wb") as f:
        pickle.dump(data_local, f)
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("\n✅ Phase 2 complete.\n")
    return gat, data_local, le


if __name__ == "__main__":
    run_phase2()
