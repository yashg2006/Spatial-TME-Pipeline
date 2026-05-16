"""
Phase 2: GNN Model Development & Training (POLISHED)
- Addresses class imbalance via weighted loss and label merging.
- Multi-scale graph evaluation (Local, Mid, Long).
- Comprehensive comparison: GAT vs GCN vs SAGE vs Random Forest.
- Early stopping, learning rate scheduling, and stratified splits.
- Reproducibility ensured via fixed seeds.
"""

import os
import pickle
import warnings
import logging
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# ── Setup ──────────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHASE1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger.info(f"Using device: {DEVICE}")

# ── Model Architectures ────────────────────────────────────────────────────────

class GCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_cls, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, hidden_ch)
        self.conv3 = GCNConv(hidden_ch, n_cls)
        self.bn1 = nn.BatchNorm1d(hidden_ch)
        self.bn2 = nn.BatchNorm1d(hidden_ch)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class SpatialTME_GAT(nn.Module):
    """GAT with multi-head attention and support for weight extraction."""
    def __init__(self, in_ch, hidden_ch, n_cls, heads=4, dropout=0.4):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden_ch, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_ch*heads, hidden_ch, heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden_ch*heads, n_cls, heads=1, dropout=dropout, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_ch * heads)
        self.bn2 = nn.BatchNorm1d(hidden_ch * heads)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attn=False):
        if return_attn:
            x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = self.bn1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
            x = self.bn2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1), (attn1, attn2)
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_cls, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, hidden_ch)
        self.conv3 = SAGEConv(hidden_ch, n_cls)
        self.bn1 = nn.BatchNorm1d(hidden_ch)
        self.bn2 = nn.BatchNorm1d(hidden_ch)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# ── Data Preparation ───────────────────────────────────────────────────────────

def prepare_labels(adata, strategy="merge_immune"):
    """Handle class imbalance by merging or excluding rare classes."""
    logger.info(f"Label strategy: {strategy}")
    counts = adata.obs['tme_label'].value_counts()
    logger.info(f"Original distribution:\n{counts.to_string()}")
    
    adata.obs['tme_label_clean'] = adata.obs['tme_label'].astype(str)
    
    if strategy == "merge_immune":
        adata.obs.loc[adata.obs['tme_label_clean'] == 'immune', 'tme_label_clean'] = 'stroma'
    elif strategy == "exclude_immune":
        adata = adata[adata.obs['tme_label_clean'] != 'immune'].copy()
    elif strategy == "binary":
        adata.obs.loc[adata.obs['tme_label_clean'].isin(['stroma', 'immune']), 'tme_label_clean'] = 'non_tumor'
    
    label_col = 'tme_label_clean'
    logger.info(f"Cleaned distribution:\n{adata.obs[label_col].value_counts().to_string()}")
    return adata, label_col

def build_pyg_data(adata, graph_key="spatial_local", label_col="tme_label_clean"):
    """Build PyG Data object using weighted graphs if available."""
    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)
    
    # Try to load weighted connectivity first
    weighted_key = f"{graph_key}_weighted"
    conn_key = weighted_key if weighted_key in adata.obsp else f"{graph_key}_connectivities"
    
    if conn_key not in adata.obsp:
        raise KeyError(f"Graph key {conn_key} not found in adata.obsp. Run Phase 1 first.")
        
    edge_index, edge_weight = from_scipy_sparse_matrix(adata.obsp[conn_key])
    edge_weight = edge_weight.float()
    
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(adata.obs[label_col].values), dtype=torch.long)
    
    # Compute class weights for balanced loss
    weights = compute_class_weight('balanced', classes=np.unique(y.numpy()), y=y.numpy())
    class_weights = torch.FloatTensor(weights)
    
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    data.label_names = list(le.classes_)
    data.class_weights = class_weights
    
    logger.info(f"Graph {graph_key}: {data.num_nodes} nodes, {data.num_edges} edges ({'weighted' if 'weighted' in conn_key else 'binary'})")
    return data, le

def create_splits(data, train_ratio=0.7, val_ratio=0.15):
    """Create stratified train/val/test splits."""
    indices = np.arange(data.num_nodes)
    y_np = data.y.numpy()
    
    # Split 1: (Train+Val) and Test
    tv_idx, test_idx = train_test_split(indices, test_size=1-train_ratio-val_ratio, stratify=y_np, random_state=RANDOM_SEED)
    
    # Split 2: Train and Val
    val_rel_size = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(tv_idx, test_size=val_rel_size, stratify=y_np[tv_idx], random_state=RANDOM_SEED)
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[train_idx], val_mask[val_idx], test_mask[test_idx] = True, True, True
    return train_mask, val_mask, test_mask

# ── Training ───────────────────────────────────────────────────────────────────

def train_step(model, data, optimizer, mask, class_weights=None):
    model.train()
    optimizer.zero_grad()
    
    # Pass edge_weight if model supports it (GCN)
    if isinstance(model, GCN):
        out = model(data.x, data.edge_index, data.edge_weight)
    else:
        out = model(data.x, data.edge_index)
        
    loss = F.nll_loss(out[mask], data.y[mask], weight=class_weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    if isinstance(model, GCN):
        out = model(data.x, data.edge_index, data.edge_weight)
    else:
        out = model(data.x, data.edge_index)
        
    pred = out.argmax(dim=1)
    y_true, y_pred = data.y[mask].cpu().numpy(), pred[mask].cpu().numpy()
    
    acc = (pred[mask] == data.y[mask]).float().mean().item()
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1_macro, y_pred, y_true

def train_model(model, data, train_mask, val_mask, test_mask, n_epochs=200, name="model"):
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    class_weights = data.class_weights.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
    
    best_val_f1, best_state, best_epoch = 0, None, 0
    patience, counter = 20, 0
    history = {'train_f1': [], 'val_f1': [], 'test_f1': [], 'loss': []}
    
    for epoch in range(1, n_epochs + 1):
        loss = train_step(model, data, optimizer, train_mask, class_weights)
        _, tr_f1, _, _ = evaluate(model, data, train_mask)
        _, val_f1, _, _ = evaluate(model, data, val_mask)
        _, te_f1, te_pred, te_true = evaluate(model, data, test_mask)
        
        scheduler.step(val_f1)
        history['loss'].append(loss); history['train_f1'].append(tr_f1)
        history['val_f1'].append(val_f1); history['test_f1'].append(te_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            
        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"{name} | Ep {epoch:3d} | Loss: {loss:.3f} | Val F1: {val_f1:.3f} | Test F1: {te_f1:.3f}")
        
        if counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break
            
    model.load_state_dict(best_state)
    _, _, final_pred, final_true = evaluate(model, data, test_mask)
    
    return {
        'name': name, 'model': model.cpu(), 'best_state': best_state,
        'history': history, 'test_f1': f1_score(final_true, final_pred, average='macro'),
        'test_pred': final_pred, 'test_true': final_true, 'label_names': data.label_names
    }

# ── Baselines & Visualization ──────────────────────────────────────────────────

def run_rf_baseline(adata, label_col):
    logger.info("Running Random Forest Baseline...")
    X, y = adata.obsm["X_pca"], LabelEncoder().fit_transform(adata.obs[label_col].values)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
        rf.fit(X[tr], y[tr])
        scores.append(f1_score(y[te], rf.predict(X[te]), average='macro'))
    return np.mean(scores), np.std(scores)

def plot_results(results, output_dir):
    # Training curves
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.plot(r['history']['val_f1'], label=r['name'], alpha=0.7)
    plt.title('Validation F1-Macro per Epoch'); plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "training_curves.png"), bbox_inches='tight')
    
    # Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    names = [r['name'] for r in results]
    f1s = [r['test_f1'] for r in results]
    sns.barplot(x=names, y=f1s, palette='viridis')
    plt.title('Model Comparison: Test F1-Macro')
    plt.xticks(rotation=45); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, "model_comparison_bar.png"), bbox_inches='tight')

# ── Main ───────────────────────────────────────────────────────────────────────

def run_phase2():
    logger.info("Starting Phase 2 Model Training...")
    adata = sc.read_h5ad(os.path.join(PHASE1_DIR, "adata_phase1.h5ad"))
    adata, label_col = prepare_labels(adata, strategy="merge_immune")
    
    in_ch, n_cls = adata.obsm["X_pca"].shape[1], adata.obs[label_col].nunique()
    hid = 128
    
    # Graphs
    data_local, le = build_pyg_data(adata, "spatial_local", label_col)
    data_mid, _ = build_pyg_data(adata, "spatial_mid", label_col)
    data_long, _ = build_pyg_data(adata, "spatial_long", label_col)
    
    train_mask, val_mask, test_mask = create_splits(data_local)
    
    # Baseline
    rf_mean, rf_std = run_rf_baseline(adata, label_col)
    logger.info(f"RF Baseline F1-Macro: {rf_mean:.4f} ± {rf_std:.4f}")
    
    all_results = []
    graph_configs = [("Local", data_local), ("Mid", data_mid), ("Long", data_long)]
    
    for g_name, g_data in graph_configs:
        logger.info(f"\n--- Training on {g_name} Graph ---")
        all_results.append(train_model(SpatialTME_GAT(in_ch, hid, n_cls), g_data, train_mask, val_mask, test_mask, name=f"GAT-{g_name}"))
        all_results.append(train_model(GCN(in_ch, hid, n_cls), g_data, train_mask, val_mask, test_mask, name=f"GCN-{g_name}"))
        all_results.append(train_model(GraphSAGE(in_ch, hid, n_cls), g_data, train_mask, val_mask, test_mask, name=f"SAGE-{g_name}"))
    
    # Results Table
    df = pd.DataFrame([{ 'Model': r['name'], 'F1-Macro': r['test_f1'] } for r in all_results])
    df.loc[len(df)] = {'Model': 'RandomForest', 'F1-Macro': rf_mean}
    df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n" + df.sort_values('F1-Macro', ascending=False).to_string(index=False))
    
    # Save Best
    best_model = max(all_results, key=lambda x: x['test_f1'])
    best_gat = max([r for r in all_results if "GAT" in r['name']], key=lambda x: x['test_f1'])
    
    torch.save(best_model['best_state'], os.path.join(MODEL_DIR, "best_model.pt"))
    torch.save(best_gat['best_state'], os.path.join(MODEL_DIR, "best_gat_model.pt"))
    
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f: pickle.dump(le, f)
    with open(os.path.join(OUTPUT_DIR, "splits.pkl"), "wb") as f: 
        pickle.dump({'train': train_mask.numpy(), 'val': val_mask.numpy(), 'test': test_mask.numpy()}, f)
    
    plot_results(all_results, OUTPUT_DIR)
    logger.info(f"🏆 Best Model: {best_model['name']} (F1: {best_model['test_f1']:.4f})")
    return best_model

if __name__ == "__main__":
    run_phase2()
