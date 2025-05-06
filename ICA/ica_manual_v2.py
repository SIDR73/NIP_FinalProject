'''
Manual ICA Implementation
Introduction to Neuro-Image Processing Final Project
Sidharth Raghavan
'''

# import modules
import os
import numpy as np
import nibabel as nib
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from nilearn import plotting, image
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ICA components to extract
n_components = 20

# directories
base_dir = "Preprocessed_MRI"
output_dir = "ICA_Results"
os.makedirs(output_dir, exist_ok=True)

'''
WHS atlas directory and files:

Kleven, H., Bjerke, I.E., Clascá, F. et al. Waxholm Space atlas of the rat brain: 
a 3D atlas supporting data analysis and integration. Nat Methods 20, 1822–1829 (2023). DOI: 10.1038/s41592-023-02034-3
'''

atlas_dir = "WHS_SD_rat_atlas_v4_pack"
atlas_label_file = os.path.join(atlas_dir, "WHS_SD_rat_atlas_v4.label")
atlas_nifti_file = os.path.join(atlas_dir, "WHS_SD_rat_atlas_v4.nii.gz")
atlas_fa_file = os.path.join(atlas_dir, "WHS_SD_rat_FA_color_v1.01.nii.gz")
atlas_t2_file = os.path.join(atlas_dir, "WHS_SD_rat_T2star_v1.01.nii.gz")

# load the WHS atlas data
def load_atlas():
    try:
        # atlas NIfTI file
        atlas_img = nib.load(atlas_nifti_file)
        atlas_data = atlas_img.get_fdata()
        
        # label file
        atlas_labels = {}
        try:
            with open(atlas_label_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                index = int(parts[0])
                                name = ' '.join(parts[1:])
                                atlas_labels[index] = name
                            except ValueError:
                                continue
        except Exception as label_error:
            print(f"Warning: Error parsing label file: {label_error}")
        
        # template files for visualization
        t2_img = nib.load(atlas_t2_file)
        fa_img = nib.load(atlas_fa_file)
        
        print(f"Loaded WHS rat atlas with {len(atlas_labels)} regions")
        
        # dimensions between atlas and template
        if atlas_img.shape != t2_img.shape:
            print(f"Warning: Atlas dimensions {atlas_img.shape} don't match T2 template dimensions {t2_img.shape}")
            print("Will attempt to resample for consistent visualization")
            # resample atlas to match T2 template
            atlas_img = image.resample_to_img(atlas_img, t2_img, interpolation='nearest')
            atlas_data = atlas_img.get_fdata()
        
        return {
            'atlas_img': atlas_img,
            'atlas_data': atlas_data,
            'labels': atlas_labels,
            't2_img': t2_img,
            'fa_img': fa_img
        }
    
    except Exception as e:
        print(f"Error loading atlas data: {e}")
        return None

# load ICA components
def load_ica_components(subject_id):
    ica_dir = os.path.join(output_dir, f"sub-{subject_id}")
    
    components = []
    for i in range(1, n_components + 1):
        comp_file = os.path.join(ica_dir, f"component_{i}.nii.gz")
        if os.path.exists(comp_file):
            comp_img = nib.load(comp_file)
            components.append(comp_img)
    
    time_courses_file = os.path.join(ica_dir, "time_courses.npy")
    time_courses = None
    if os.path.exists(time_courses_file):
        time_courses = np.load(time_courses_file)
    
    print(f"Loaded {len(components)} ICA components for subject {subject_id}")
    return components, time_courses

# register ICA components to atlas space
def register_component_to_atlas(component_img, atlas_img):
    # resample component to atlas space
    resampled_img = image.resample_to_img(
        component_img, 
        atlas_img,
        interpolation='linear'
    )
    return resampled_img

# overlay ICA components on atlas
def create_atlas_overlays(components, atlas_data, subject_id):
    overlay_dir = os.path.join(output_dir, f"sub-{subject_id}", "atlas_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    
    atlas_img = atlas_data['atlas_img']
    t2_img = atlas_data['t2_img']
    atlas_labels = atlas_data['labels']
    
    # z-score for ICA components (thresholding)
    threshold = 1.5
    
    for i, comp_img in enumerate(components):
        comp_num = i + 1
        print(f"Creating overlay for component {comp_num}/{len(components)}")
        
        # register component to atlas space
        registered_comp = register_component_to_atlas(comp_img, t2_img)
        
        reg_file = os.path.join(overlay_dir, f"component_{comp_num}_registered.nii.gz")
        nib.save(registered_comp, reg_file)
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle(f"Component {comp_num} on WHS Atlas", fontsize=16)
        
        t2_data = t2_img.get_fdata()
        comp_data = registered_comp.get_fdata()
        
        x_slices = np.linspace(0, t2_data.shape[0]-1, 5).astype(int)
        y_slices = np.linspace(0, t2_data.shape[1]-1, 5).astype(int)
        z_slices = np.linspace(0, t2_data.shape[2]-1, 5).astype(int)
        
        # Plot sagittal slices
        for j, x in enumerate(x_slices):
            ax = axes[0, j]
            ax.imshow(np.rot90(t2_data[x, :, :]), cmap='gray')

            comp_slice = np.rot90(comp_data[x, :, :])
            masked_comp = np.ma.masked_where(np.abs(comp_slice) < threshold, comp_slice)
            ax.imshow(masked_comp, cmap='hot', alpha=0.7)
            ax.set_title(f"X={x}")
            ax.axis('off')
        
        # plot coronal slices
        for j, y in enumerate(y_slices):
            ax = axes[1, j]
            ax.imshow(np.rot90(t2_data[:, y, :]), cmap='gray')

            comp_slice = np.rot90(comp_data[:, y, :])
            masked_comp = np.ma.masked_where(np.abs(comp_slice) < threshold, comp_slice)
            ax.imshow(masked_comp, cmap='hot', alpha=0.7)
            ax.set_title(f"Y={y}")
            ax.axis('off')
        
        # plot axial slices 
        for j, z in enumerate(z_slices):
            ax = axes[2, j]
            ax.imshow(t2_data[:, :, z], cmap='gray')
            comp_slice = comp_data[:, :, z]
            masked_comp = np.ma.masked_where(np.abs(comp_slice) < threshold, comp_slice)
            ax.imshow(masked_comp, cmap='hot', alpha=0.7)
            ax.set_title(f"Z={z}")
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(overlay_dir, f"component_{comp_num}_overlay.png"), dpi=150)
        plt.close()
        
        # 3d projections
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(f"Component {comp_num} - 3D Projections", fontsize=16)
        
        # max intensity projections
        ax1 = fig.add_subplot(131)
        max_proj_x = np.max(comp_data, axis=0)
        masked_proj_x = np.ma.masked_where(np.abs(max_proj_x) < threshold, max_proj_x)
        ax1.imshow(np.rot90(t2_data[:, :, t2_data.shape[2]//2]), cmap='gray')
        ax1.imshow(np.rot90(masked_proj_x), cmap='hot', alpha=0.7)
        ax1.set_title("Sagittal Projection")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(132)
        max_proj_y = np.max(comp_data, axis=1)
        masked_proj_y = np.ma.masked_where(np.abs(max_proj_y) < threshold, max_proj_y)
        ax2.imshow(np.rot90(t2_data[:, t2_data.shape[1]//2, :]), cmap='gray')
        ax2.imshow(np.rot90(masked_proj_y), cmap='hot', alpha=0.7)
        ax2.set_title("Coronal Projection")
        ax2.axis('off')
        
        ax3 = fig.add_subplot(133)
        max_proj_z = np.max(comp_data, axis=2)
        masked_proj_z = np.ma.masked_where(np.abs(max_proj_z) < threshold, max_proj_z)
        ax3.imshow(t2_data[t2_data.shape[0]//2, :, :].T, cmap='gray')
        ax3.imshow(masked_proj_z.T, cmap='hot', alpha=0.7)
        ax3.set_title("Axial Projection")
        ax3.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(overlay_dir, f"component_{comp_num}_projections.png"), dpi=150)
        plt.close()
        
        comp_abs = np.abs(comp_data)
        max_value = np.max(comp_abs)
        if max_value > 0:
            max_coords = np.unravel_index(np.argmax(comp_abs), comp_abs.shape)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Component {comp_num} - Maximum Intensity Regions", fontsize=16)
            
            # sagittal
            axes[0].imshow(np.rot90(t2_data[max_coords[0], :, :]), cmap='gray')
            comp_slice = np.rot90(comp_data[max_coords[0], :, :])
            masked_comp = np.ma.masked_where(np.abs(comp_slice) < threshold, comp_slice)
            axes[0].imshow(masked_comp, cmap='hot', alpha=0.7)
            axes[0].set_title(f"Sagittal (X={max_coords[0]})")
            axes[0].axis('off')
            
            # coronal
            axes[1].imshow(np.rot90(t2_data[:, max_coords[1], :]), cmap='gray')
            comp_slice = np.rot90(comp_data[:, max_coords[1], :])
            masked_comp = np.ma.masked_where(np.abs(comp_slice) < threshold, comp_slice)
            axes[1].imshow(masked_comp, cmap='hot', alpha=0.7)
            axes[1].set_title(f"Coronal (Y={max_coords[1]})")
            axes[1].axis('off')
            
            # axial
            axes[2].imshow(t2_data[:, :, max_coords[2]], cmap='gray')
            comp_slice = comp_data[:, :, max_coords[2]]
            masked_comp = np.ma.masked_where(np.abs(comp_slice) < threshold, comp_slice)
            axes[2].imshow(masked_comp, cmap='hot', alpha=0.7)
            axes[2].set_title(f"Axial (Z={max_coords[2]})")
            axes[2].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(overlay_dir, f"component_{comp_num}_max_intensity.png"), dpi=150)
            plt.close()
        
        analyze_region_overlap(registered_comp, atlas_data, comp_num, overlay_dir, threshold)
        
# analyze which atlas regions overlap with ICA components
def analyze_region_overlap(comp_img, atlas_data, comp_num, output_dir, threshold=1.5):

    atlas_img = atlas_data['atlas_img']
    atlas_volume = atlas_img.get_fdata()
    labels = atlas_data['labels']
    
    comp_data = comp_img.get_fdata()
    
    # threshold component data
    thresholded_comp = np.abs(comp_data) > threshold
    
    #dictionary for region overlap
    region_overlap = {}
    
    unique_regions = np.unique(atlas_volume).astype(int)
    
    # overlap for each region
    for region_idx in unique_regions:
        if region_idx == 0:
            continue
            
        region_mask = (atlas_volume == region_idx)
        
        overlap_voxels = np.sum(np.logical_and(region_mask, thresholded_comp))
        region_size = np.sum(region_mask)
        
        if region_size > 0:
            overlap_percentage = (overlap_voxels / region_size) * 100
            
            if overlap_voxels > 0:
                region_name = labels.get(region_idx, f"Region {region_idx}")
                region_overlap[region_name] = {
                    'overlap_voxels': int(overlap_voxels),
                    'region_size': int(region_size),
                    'overlap_percentage': float(overlap_percentage)
                }
    
    sorted_regions = sorted(
        region_overlap.items(), 
        key=lambda x: x[1]['overlap_percentage'], 
        reverse=True
    )
    
    #save overlap data to csv
    csv_file = os.path.join(output_dir, f"component_{comp_num}_region_overlap.csv")
    with open(csv_file, 'w') as f:
        f.write("Region,Overlap Voxels,Region Size,Overlap Percentage\n")
        for region_name, data in sorted_regions:
            f.write(f"{region_name},{data['overlap_voxels']},{data['region_size']},{data['overlap_percentage']:.2f}\n")
    
    # plot of top 10 regions
    if len(sorted_regions) > 0:
        top_regions = sorted_regions[:min(10, len(sorted_regions))]
        
        plt.figure(figsize=(10, 6))
        region_names = [r[0] if len(r[0]) < 30 else r[0][:27]+"..." for r in top_regions]
        percentages = [r[1]['overlap_percentage'] for r in top_regions]
        
        plt.barh(region_names, percentages, color='steelblue')
        plt.xlabel('Overlap Percentage (%)')
        plt.ylabel('Brain Region')
        plt.title(f'Top Regions Overlapping with Component {comp_num}')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"component_{comp_num}_region_overlap.png"), dpi=150)
        plt.close()
    
    return sorted_regions

def main(subject_id="09"):
    #load atlas
    print("Loading WHS atlas:")
    atlas_data = load_atlas()
    if not atlas_data:
        print("Failed to load atlas. Exiting.")
        return
    
    #load ica components
    print(f"Loading ICA components for subject {subject_id}...")
    components, time_courses = load_ica_components(subject_id)
    if not components:
        print("No ICA components found. Exiting.")
        return
    
    print("Creating atlas overlays:")
    create_atlas_overlays(components, atlas_data, subject_id)
    
    print("Atlas visualization complete")
    print(f"Results saved to: {os.path.join(output_dir, f'sub-{subject_id}', 'atlas_overlays')}")

if __name__ == "__main__":
    main("09")