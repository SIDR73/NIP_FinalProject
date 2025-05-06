'''
Sklearn ICA Implementation
Introduction to Neuro-Image Processing Final Project
Sidharth Raghavan
'''

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

groups = {
    'SHAM': ['09', '14'],
    'INC_MUSC': ['03', '15', '16', '18'],
    'INC_BIC': ['04', '07', '08'],
    'INC_SALINE': ['01', '17', '22']
}
n_components = 20
base_dir = "Preprocessed_MRI"
output_dir = "ICA_Results_v2"
os.makedirs(output_dir, exist_ok=True)
atlas_dir = "WHS_SD_rat_atlas_v4_pack"
atlas_label_file = os.path.join(atlas_dir, "WHS_SD_rat_atlas_v4.label")
atlas_nifti_file = os.path.join(atlas_dir, "WHS_SD_rat_atlas_v4.nii.gz")
atlas_t2_file = os.path.join(atlas_dir, "WHS_SD_rat_T2star_v1.01.nii.gz")

# load atlas
def load_atlas():
    atlas_img = nib.load(atlas_nifti_file)
    atlas_data = atlas_img.get_fdata()
    atlas_labels = {}
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
    t2_img = nib.load(atlas_t2_file)

    if atlas_img.shape != t2_img.shape:
        atlas_img = image.resample_to_img(atlas_img, t2_img, interpolation='nearest')
        atlas_data = atlas_img.get_fdata()
    return {'atlas_img': atlas_img, 'atlas_data': atlas_data, 'labels': atlas_labels, 't2_img': t2_img}

# load and preprocess
def load_mri_data(subject_id):
    file_path = os.path.join(base_dir, f"sub-{subject_id}", "func", "preprocessed", "mc_func.nii.gz")
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    dims = data.shape
    n_timepoints = dims[3]
    data_2d = data.reshape(-1, n_timepoints).T
    data_2d = np.nan_to_num(data_2d)
    std_map = np.std(data, axis=3)
    mask = std_map > np.percentile(std_map, 15)
    mask_flat = mask.flatten()
    data_2d_masked = data_2d[:, mask_flat]
    data_2d_masked = (data_2d_masked - np.mean(data_2d_masked, axis=0)) / (np.std(data_2d_masked, axis=0) + 1e-10)

    # temporal filtering
    from scipy.signal import butter, filtfilt
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data, axis=0)
        return y
    TR = 1.0
    fs = 1.0/TR
    cutoff = 0.1
    try:
        data_2d_filtered = butter_lowpass_filter(data_2d_masked, cutoff, fs)
    except Exception:
        data_2d_filtered = data_2d_masked
    return {
        'data_2d': data_2d_filtered,
        'original_shape': dims[0:3],
        'affine': affine,
        'img': img,
        'mask': mask
    }

# ica implementation with inbuilt whitening
def apply_ica(data, n_components=20):
    ica = FastICA(n_components=n_components, random_state=42, max_iter=2000, tol=1e-5, whiten='unit-variance')
    try:
        components = ica.fit_transform(data)
    except Exception:
        ica = FastICA(n_components=n_components, random_state=42, max_iter=5000, tol=1e-4, whiten=True)
        components = ica.fit_transform(data)
    mixing_matrix = ica.mixing_
    component_maps = ica.components_
    component_var = np.var(component_maps, axis=1)
    idx_sort = np.argsort(-component_var)
    component_maps = component_maps[idx_sort]
    mixing_matrix = mixing_matrix[:, idx_sort]
    return {
        'mixing_matrix': mixing_matrix,
        'component_maps': component_maps,
        'ica_model': ica,
        'variance': component_var[idx_sort]
    }

# reconstruct component maps
def reconstruct_component_maps(component_maps, original_shape, mask=None):
    n_components = component_maps.shape[0]
    reconstructed_maps = []
    for i in range(n_components):
        if mask is not None:
            component_3d = np.zeros(np.prod(original_shape))
            mask_flat = mask.flatten()
            component_3d[mask_flat] = component_maps[i, :]
            component_3d = component_3d.reshape(original_shape)
        else:
            component_3d = component_maps[i].reshape(original_shape)
        from scipy.ndimage import gaussian_filter
        component_3d = gaussian_filter(component_3d, sigma=0.7)
        mean = np.mean(component_3d)
        std = np.std(component_3d)
        if std > 0:
            component_3d = (component_3d - mean) / std
        reconstructed_maps.append(component_3d)
    return reconstructed_maps

# register component maps to atlas
def register_to_atlas(component_map, subject_affine, atlas_info):
    comp_img = nib.Nifti1Image(component_map, subject_affine)
    comp_data = component_map.copy()
    comp_data = (comp_data - np.mean(comp_data)) / (np.std(comp_data) + 1e-10)
    comp_img = nib.Nifti1Image(comp_data, subject_affine)
    resampled_img = image.resample_to_img(comp_img, atlas_info['t2_img'], interpolation='linear')
    return resampled_img

#label components with atlas
def label_components_with_atlas(component_img, atlas_img, atlas_labels, threshold=1.5):
    comp_data = component_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    comp_mask = np.abs(comp_data) > threshold
    region_scores = []
    for region_idx, region_name in atlas_labels.items():
        region_mask = atlas_data == region_idx
        overlap = np.sum(comp_mask & region_mask)
        if overlap > 0:
            region_scores.append((region_name, overlap))
    region_scores.sort(key=lambda x: -x[1])
    return region_scores

def save_component_maps(component_maps, affine, subject_id, atlas_info=None):
    subject_output_dir = os.path.join(output_dir, f"sub-{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)
    for i, component_map in enumerate(component_maps):
        nii_img = nib.Nifti1Image(component_map, affine)
        if atlas_info:
            nii_img = register_to_atlas(component_map, affine, atlas_info)
        output_file = os.path.join(subject_output_dir, f"component_{i+1}.nii.gz")
        nib.save(nii_img, output_file)
    print(f"Saved {len(component_maps)} component maps for subject {subject_id}")

def visualize_component_maps(component_maps, affine, subject_id, atlas_info=None):
    subject_output_dir = os.path.join(output_dir, f"sub-{subject_id}", "figures")
    os.makedirs(subject_output_dir, exist_ok=True)
    for i, component_map in enumerate(component_maps):
        nii_img = nib.Nifti1Image(component_map, affine)
        if atlas_info:
            nii_img = register_to_atlas(component_map, affine, atlas_info)
        component_values = nii_img.get_fdata()
        abs_values = np.abs(component_values)
        abs_values = abs_values[abs_values > 0]
        threshold = np.percentile(abs_values, 80) if len(abs_values) > 0 else 0.5
        threshold = max(0.8, min(threshold, 2.0))
        output_file = os.path.join(subject_output_dir, f"component_{i+1}.png")
        fig = plt.figure(figsize=(10, 4))
        display = plotting.plot_stat_map(
            nii_img,
            bg_img=atlas_info['t2_img'] if atlas_info else None,
            display_mode='ortho',
            title=f"Subject {subject_id} - Component {i+1}",
            colorbar=True,
            threshold=threshold,
            vmax=3.0,
            cmap='RdBu_r'
        )
        if atlas_info:
            try:
                display.add_contours(
                    atlas_info['atlas_img'],
                    levels=list(range(1, 10)),
                    colors='white',
                    linewidths=0.5,
                    alpha=0.7
                )
            except Exception as e:
                print(f"Warning: Could not add contours: {e}")
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close(fig)

def run_ica_pipeline(subject_id, atlas_info):
    mri = load_mri_data(subject_id)
    ica_result = apply_ica(mri['data_2d'], n_components=n_components)
    component_maps = reconstruct_component_maps(ica_result['component_maps'], mri['original_shape'], mask=mri['mask'])
    save_component_maps(component_maps, mri['affine'], subject_id, atlas_info=atlas_info)
    visualize_component_maps(component_maps, mri['affine'], subject_id, atlas_info=atlas_info)
    
    #register component maps to atlas and label them
    overlap_results = []
    for i, component_map in enumerate(component_maps):
        nii_img = register_to_atlas(component_map, mri['affine'], atlas_info)
        overlap_info = label_components_with_atlas(nii_img, atlas_info['atlas_img'], atlas_info['labels'])
        if overlap_info:
            print(f"Subject {subject_id} - Component {i+1} overlaps most with: {overlap_info[:3]}")
        overlap_results.append({'component': i+1, 'top_regions': overlap_info[:3]})
    subject_output_dir = os.path.join(output_dir, f"sub-{subject_id}")
    df = pd.DataFrame(overlap_results)
    df.to_csv(os.path.join(subject_output_dir, "ica_atlas_overlap.csv"), index=False)

if __name__ == "__main__":
    atlas_info = load_atlas()
    for group, subjects in groups.items():
        for subject_id in subjects:
            print(f"\n=== Processing subject {subject_id} ({group}) ===")
            run_ica_pipeline(subject_id, atlas_info)