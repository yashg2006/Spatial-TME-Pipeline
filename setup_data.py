"""
Data Setup Script
Creates the correct directory structure that scanpy's read_visium() expects,
by symlinking/copying downloaded files into place.
"""

import os
import shutil

BASE    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "V1_Breast_Cancer_Block_A_Section_1")
DST_DIR = os.path.join(BASE, "data", "section1")

os.makedirs(DST_DIR, exist_ok=True)

# read_visium() expects:
#   <dir>/
#     filtered_feature_bc_matrix.h5   OR   filtered_feature_bc_matrix/
#     spatial/
#       tissue_positions_list.csv
#       scalefactors_json.json
#       tissue_hires_image.png
#       tissue_lowres_image.png

PREFIX = "V1_Breast_Cancer_Block_A_Section_1_"

# 1. Filtered H5 file
h5_src = os.path.join(RAW_DIR, f"{PREFIX}filtered_feature_bc_matrix.h5")
h5_dst = os.path.join(DST_DIR, "filtered_feature_bc_matrix.h5")
if os.path.exists(h5_src) and not os.path.exists(h5_dst):
    shutil.copy2(h5_src, h5_dst)
    print(f"Copied: {h5_dst}")

# 2. Spatial folder
# Raw download extracts to: RAW_DIR/V1_..._spatial/spatial/
spatial_candidates = [
    os.path.join(RAW_DIR, f"{PREFIX}spatial", "spatial"),
    os.path.join(RAW_DIR, f"{PREFIX}spatial"),
    os.path.join(RAW_DIR, "spatial"),
]
spatial_dst = os.path.join(DST_DIR, "spatial")

if not os.path.exists(spatial_dst):
    for cand in spatial_candidates:
        if os.path.isdir(cand):
            shutil.copytree(cand, spatial_dst)
            print(f"Copied spatial dir from: {cand}")
            break
    else:
        print("WARNING: Could not find spatial folder. Check RAW_DIR structure.")

# 3. Verify
missing = []
required = [
    os.path.join(DST_DIR, "filtered_feature_bc_matrix.h5"),
    os.path.join(DST_DIR, "spatial", "tissue_positions_list.csv"),
    os.path.join(DST_DIR, "spatial", "scalefactors_json.json"),
]
for p in required:
    if not os.path.exists(p):
        missing.append(p)

if missing:
    print("\n⚠️  Missing files:")
    for m in missing:
        print(f"   {m}")
else:
    print("\n✅ Data directory ready. Run: python run_pipeline.py")
