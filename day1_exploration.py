import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Define the base directory where data is stored
    data_dir = "V1_Breast_Cancer_Block_A_Section_1"
    
    # scanpy's read_visium function typically expects a folder containing:
    # 1. A 'spatial' folder (with tissue_hires_image.png, scalefactors_json.json, tissue_positions_list.csv)
    # 2. The filtered_feature_bc_matrix.h5 file
    # We will rearrange the files slightly so read_visium works seamlessly.
    
    spatial_src = os.path.join(data_dir, "V1_Breast_Cancer_Block_A_Section_1_spatial", "spatial")
    spatial_dest = os.path.join(data_dir, "spatial")
    
    if os.path.exists(spatial_src) and not os.path.exists(spatial_dest):
        os.rename(spatial_src, spatial_dest)
        
    h5_file = "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5"
    h5_src = os.path.join(data_dir, h5_file)
    h5_dest = os.path.join(data_dir, "filtered_feature_bc_matrix.h5")
    
    if os.path.exists(h5_src) and not os.path.exists(h5_dest):
        os.rename(h5_src, h5_dest)
        
    print("Loading Visium data...")
    try:
        # Load the data
        adata = sc.read_visium(data_dir)
        adata.var_names_make_unique()
        print(f"Data loaded successfully! Structure: {adata}")
        
        # Day 1 exploration: plot H&E image with some spots
        print("Saving spatial plot to 'initial_spatial_plot.png'...")
        sc.pl.spatial(adata, show=False)
        plt.savefig("initial_spatial_plot.png", dpi=300, bbox_inches='tight')
        print("Exploration complete. Check 'initial_spatial_plot.png'.")
        
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    main()
