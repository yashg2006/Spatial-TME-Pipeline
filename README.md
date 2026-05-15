# Spatial TME Pipeline

A rigorous, reproducible bioinformatics pipeline for analyzing spatial tumor microenvironments (TME) using Graph Neural Networks (GNNs).

## Project Overview
This project implements a multi-phase analysis of 10x Genomics Visium datasets:
1. **Phase 1: Data Preparation** - Advanced QC, normalization, and weighted spatial graph construction.
2. **Phase 2: Model Development** - Training GNN models (GAT, GraphSAGE, GCN) for TME classification.
3. **Phase 3: Explainability** - Generating consensus gene sets using GNNExplainer and Attention mechanisms.
4. **Phase 4: Validation** - Biological validation via GSEA and marker overlap analysis.
5. **Phase 5: Generalization** - Testing model robustness on adjacent tissue sections.

## Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the pipeline phases sequentially:
```powershell
python run_pipeline.py --phase 1
python run_pipeline.py --phase 2
```

## Key Technologies
- `scanpy` & `squidpy` for spatial transcriptomics.
- `PyTorch Geometric` for GNN implementation.
- `captum` for model explainability.
