'''
Manual ICA Implementation with Proper Whitening and Masking
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

# Load the WHS atlas data
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

def run_manual_ica(subject_id, n_components=20):
    """
    Runs ICA with proper whitening and masking on a single subject
    following the protocol steps exactly
    """
    # Construct file path
    fmri_file = os.path.join(base_dir, f"sub-{subject_id}", "func", "preprocessed", "mc_func.nii.gz")
    
    if not os.path.exists(fmri_file):
        print(f"fMRI file not found: {fmri_file}")
        return
    
    print(f"Loading fMRI data for subject {subject_id}...")
    fmri_img = nib.load(fmri_file)
    fmri_data = fmri_img.get_fdata()
    
    x, y, z, t = fmri_data.shape
    print(f"fMRI dimensions: {fmri_data.shape}")
    
    # create brain mask
    # simple intensity threshold for brain masking
    # take mean across time dimension
    mean_image = np.mean(fmri_data, axis=3)
    
    # threshold to create binary mask
    threshold = np.mean(mean_image) * 0.8
    mask = (mean_image > threshold).astype(int)
    
    #ensure low voxel number for efficient whitening
    num_voxels = np.sum(mask)
    print(f"Initial mask contains {num_voxels} voxels")
    
    # Dynamically adjust threshold to get close to 10,000 voxels
    adjustment_attempts = 0
    while (num_voxels > 15000 or num_voxels < 8000) and adjustment_attempts < 10:
        adjustment_attempts += 1
        if num_voxels > 15000:
            threshold *= 1.1
        else:
            threshold *= 0.9
        
        mask = (mean_image > threshold).astype(int)
        num_voxels = np.sum(mask)
        print(f"Adjusted mask contains {num_voxels} voxels (threshold: {threshold:.2f})")
    
    subject_output_dir = os.path.join(output_dir, f"sub-{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)
    
    mask_img = nib.Nifti1Image(mask.astype(np.float32), fmri_img.affine)
    nib.save(mask_img, os.path.join(subject_output_dir, "brain_mask.nii.gz"))
    
    mask_indices = np.where(mask == 1)
    
    # extract time series data
    X = np.zeros((num_voxels, t))
    for i in range(num_voxels):
        x_coord, y_coord, z_coord = mask_indices[0][i], mask_indices[1][i], mask_indices[2][i]
        X[i, :] = fmri_data[x_coord, y_coord, z_coord, :]
    
    print(f"Extracted time series data matrix shape: {X.shape}")
    
    # subtract the mean from each voxel's time series
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    
    # whiten the data
    print("Computing covariance matrix:")

    # compute voxel-wise covariance matrix
    C = np.dot(X_centered, X_centered.T) / t
    
    print("Performing eigendecomposition:")
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # select top K eigenvectors
    K = n_components
    print(f"Using top {K} eigenvectors for whitening")
    top_eigenvalues = eigenvalues[:K]
    top_eigenvectors = eigenvectors[:, :K]
    
    # compute whitening matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(top_eigenvalues))
    V = np.dot(D_inv_sqrt, top_eigenvectors.T)
    
    # transform the data
    Z = np.dot(V, X_centered)
    
    print(f"Whitened data shape: {Z.shape}")
    
    # FastICA iteration
    print("Running FastICA:")
    
    # initialize unmixing matrix W randomly
    W = np.random.normal(size=(K, K))
    # orthogonalize W
    W, _ = np.linalg.qr(W)
    
    max_iter = 100
    tolerance = 1e-4
    
    #nonlinear contrast function
    def g(x):
        return np.tanh(x)
    
    def g_prime(x):
        return 1 - np.tanh(x)**2
    
    # FastICA iteration
    for iteration in range(max_iter):
        W_old = W.copy()
        
        #update each row of W
        for k in range(K):
            w = W[k, :].copy().reshape(K, 1)
            
            #compute w_new = E[Z*g(w^T*Z)] - E[g'(w^T*Z)]*w
            w_z = np.dot(w.T, Z)  # (1, T)
            gw_z = g(w_z)  # (1, T)
            g_prime_w_z = g_prime(w_z)  # (1, T)
            
            term1 = np.dot(Z, gw_z.T) / t  # (K, 1)
            term2 = np.mean(g_prime_w_z) * w  # (K, 1)
            
            w_new = term1 - term2
            
            #normalize
            w_new = w_new / np.linalg.norm(w_new)
            
            #decorrelate from other vectors (Gram-Schmidt orthogonalization)
            for j in range(k):
                w_new = w_new - np.dot(W[j, :].reshape(K, 1).T, w_new) * W[j, :].reshape(K, 1)
                w_new = w_new / np.linalg.norm(w_new)
            
            W[k, :] = w_new.ravel()
        
        # Check for convergence
        lim = np.max(np.abs(np.abs(np.diag(np.dot(W, W_old.T))) - 1))
        if lim < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
            
    print("FastICA completed")
    
    #recover spatial maps
    #compute mixing matrix A = W^(-1)
    A = np.linalg.inv(W)
    
    #reverse the whitening to obtain spatial maps
    M = np.dot(top_eigenvectors, np.dot(np.diag(np.sqrt(top_eigenvalues)), A))
    
    print(f"Spatial maps shape: {M.shape}")
    
    # normalize and sort components
    # Z-score each spatial map
    M_z = np.zeros_like(M)
    for k in range(K):
        mu_k = np.mean(M[:, k])
        sigma_k = np.std(M[:, k])
        M_z[:, k] = (M[:, k] - mu_k) / sigma_k
    
    #compute time courses
    S = np.dot(W, Z)
    
    #sort by variance of time courses
    var_s = np.var(S, axis=1)
    sort_idx = np.argsort(var_s)[::-1]
    
    #sort components by variance
    S_sorted = S[sort_idx, :]
    M_z_sorted = M_z[:, sort_idx]
    
    print(f"Components sorted by variance: {var_s[sort_idx]}")
    
    #convert back to brain space for visualization
    #initialize empty volumes for each component - using float32 to avoid int64 issues
    component_volumes = np.zeros((x, y, z, K), dtype=np.float32)
    
    #fill volumes with component values
    for k in range(K):
        volume = np.zeros((x, y, z), dtype=np.float32)
        for i in range(num_voxels):
            x_coord, y_coord, z_coord = mask_indices[0][i], mask_indices[1][i], mask_indices[2][i]
            volume[x_coord, y_coord, z_coord] = M_z_sorted[i, k]
        
        component_volumes[:, :, :, k] = volume
    
    #save data
    # spatial maps and time courses
    for k in range(K):
        # Save spatial map as NIfTI - explicitly convert to float32 to avoid int64 issues
        component_img = nib.Nifti1Image(component_volumes[:, :, :, k].astype(np.float32), fmri_img.affine)
        nib.save(component_img, os.path.join(subject_output_dir, f"component_{k+1}_spatial.nii.gz"))
        
        #save time course as CSV
        pd.DataFrame(S_sorted[k, :]).to_csv(
            os.path.join(subject_output_dir, f"component_{k+1}_timecourse.csv"), 
            header=False, index=False
        )
    
    plot_dir = os.path.join(subject_output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    for k in range(min(K, 5)):
        plt.subplot(5, 1, k+1)
        plt.plot(S_sorted[k, :])
        plt.title(f"Component {k+1} Time Course")
        plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "top5_timecourses.png"))
    
    return {
        'spatial_maps': M_z_sorted,
        'time_courses': S_sorted,
        'mask': mask,
        'mask_indices': mask_indices,
        'component_volumes': component_volumes,
        'output_dir': subject_output_dir
    }

def plot_component_overlays(component_volumes, fmri_img, subject_id, atlas=None):
    """
    Generate overlay visualizations for ICA components
    """
    subject_output_dir = os.path.join(output_dir, f"sub-{subject_id}")
    plot_dir = os.path.join(subject_output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    _, _, _, K = component_volumes.shape
    
    if atlas and 't2_img' in atlas:
        background_img = atlas['t2_img']
        use_atlas = True
    
    #plot each component
    for k in range(K):
        #ensure component is float32 to avoid int64 issues
        component_img = nib.Nifti1Image(component_volumes[:, :, :, k].astype(np.float32), fmri_img.affine)
        
        #glass brain view
        plt.figure(figsize=(10, 6))
        try:
            display = plotting.plot_glass_brain(
                component_img,
                colorbar=True,
                title=f"Component {k+1}",
                display_mode='ortho',
                threshold=1.0  # Adjust threshold as needed
            )
            plt.savefig(os.path.join(plot_dir, f"component_{k+1}_glass.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting glass brain for component {k+1}: {e}")
        
        plt.figure(figsize=(12, 4))
        try:
            # Use specific coordinates: y=-6, x=-1, z=1
            display = plotting.plot_stat_map(
                component_img,
                bg_img=background_img,
                colorbar=True,
                title=f"Component {k+1}",
                cut_coords=(-1, -6, 1),
                threshold=1.5
            )
            plt.savefig(os.path.join(plot_dir, f"component_{k+1}_overlay.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting overlay for component {k+1}: {e}")


def run_ica_pipeline(subject_ids, n_components=20):

    atlas = load_atlas()
    
    results = {}
    for subject_id in subject_ids:
        print(f"\n===== Processing subject {subject_id} =====")
        try:
            ica_results = run_manual_ica(subject_id, n_components)
            if ica_results:
                # get fMRI image
                fmri_file = os.path.join(base_dir, f"sub-{subject_id}", "func", "preprocessed", "mc_func.nii.gz")
                fmri_img = nib.load(fmri_file)
                
                plot_component_overlays(ica_results['component_volumes'], fmri_img, subject_id, atlas)
                
                results[subject_id] = ica_results
                print(f"Successfully processed subject {subject_id}")
            else:
                print(f"Failed to process subject {subject_id}")
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
    
    return results


if __name__ == "__main__":
    
    subject_ids = ["09"]
    
    results = run_ica_pipeline(subject_ids, n_components=20)
    
    print("\nICA processing completed for all subjects.")