# ICA for Brain Network Extraction from MRI Data

import os
import numpy as np
import nibabel as nib
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from nilearn import plotting
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Define the subject groups
groups = {
    'SHAM': ['09', '14'],
    'INC_MUSC': ['03', '15', '16', '18'],
    'INC_BIC': ['04', '07', '08'],
    'INC_SALINE': ['01', '17', '22']
}

# Number of ICA components to extract
n_components = 20

# Base directory for MRI data
base_dir = "Preprocessed_MRI"

# Output directory for results
output_dir = "ICA_Results"
os.makedirs(output_dir, exist_ok=True)

# Function to load and prepare MRI data
def load_mri_data(subject_id):
    """Load 4D MRI data for a given subject and prepare it for ICA."""
    file_path = os.path.join(base_dir, f"sub-{subject_id}", "func", "preprocessed", "mc_func.nii.gz")
    
    try:
        # Load the NIfTI file
        img = nib.load(file_path)
        
        # Get the 4D data array (time × x × y × z)
        data = img.get_fdata()
        
        # Get the affine transformation matrix
        affine = img.affine
        
        # Get the dimensions
        dims = data.shape
        
        print(f"Loaded MRI data for subject {subject_id}: shape {dims}")
        
        # Reshape to 2D: time × voxels
        n_timepoints = dims[3]
        data_2d = data.reshape(-1, n_timepoints).T
        
        # Remove NaN and Inf values
        data_2d = np.nan_to_num(data_2d)
        
        # Basic normalization: remove mean and scale
        data_2d = (data_2d - np.mean(data_2d, axis=0)) / (np.std(data_2d, axis=0) + 1e-10)
        
        return {
            'data_2d': data_2d,
            'original_shape': dims[0:3],
            'affine': affine,
            'img': img
        }
        
    except Exception as e:
        print(f"Error loading data for subject {subject_id}: {e}")
        return None

# Function to apply ICA to data
def apply_ica(data, n_components=20):
    """Apply Independent Component Analysis to the data."""
    print(f"Applying ICA with {n_components} components...")
    
    # Initialize FastICA
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    
    # Apply ICA to the data
    components = ica.fit_transform(data)
    
    # Get the mixing matrix
    mixing_matrix = ica.mixing_
    
    # Get the component maps
    component_maps = ica.components_
    
    return {
        'mixing_matrix': mixing_matrix,
        'component_maps': component_maps,
        'ica_model': ica
    }

# Function to reconstruct 3D component maps
def reconstruct_component_maps(component_maps, original_shape):
    """Reconstruct 3D spatial maps from the ICA components."""
    n_components = component_maps.shape[0]
    reconstructed_maps = []
    
    for i in range(n_components):
        # Reshape the component map back to 3D
        component_3d = component_maps[i].reshape(original_shape)
        reconstructed_maps.append(component_3d)
    
    return reconstructed_maps

# Function to save component maps as NIfTI files
def save_component_maps(component_maps, affine, subject_id):
    """Save the component maps as NIfTI files."""
    subject_output_dir = os.path.join(output_dir, f"sub-{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)
    
    for i, component_map in enumerate(component_maps):
        # Create a NIfTI image from the component map
        nii_img = nib.Nifti1Image(component_map, affine)
        
        # Save the image
        output_file = os.path.join(subject_output_dir, f"component_{i+1}.nii.gz")
        nib.save(nii_img, output_file)
        
    print(f"Saved {len(component_maps)} component maps for subject {subject_id}")

# Function to visualize component maps
def visualize_component_maps(component_maps, affine, subject_id):
    """Create visualizations of the component maps."""
    subject_output_dir = os.path.join(output_dir, f"sub-{subject_id}", "figures")
    os.makedirs(subject_output_dir, exist_ok=True)
    
    for i, component_map in enumerate(component_maps):
        # Create a NIfTI image from the component map
        nii_img = nib.Nifti1Image(component_map, affine)
        
        # Create the output figure path
        output_file = os.path.join(subject_output_dir, f"component_{i+1}.png")
        
        # Plot using nilearn's plotting functions
        fig = plt.figure(figsize=(10, 6))
        # For 'ortho' mode, cut_coords needs to be a list of 3 coordinates or None
        display = plotting.plot_stat_map(
            nii_img, 
            cut_coords=None,  # Let nilearn automatically determine best cut coordinates
            display_mode='ortho',
            title=f"Subject {subject_id} - Component {i+1}",
            colorbar=True
        )
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Created visualizations for {len(component_maps)} components for subject {subject_id}")

# Function to extract group-level components with mouse brain adjustments
def extract_group_components(group_name, subject_ids, n_components=20):
    """Extract group-level ICA components by combining data from multiple subjects."""
    print(f"\nProcessing {group_name} group with subjects: {subject_ids}")
    
    # First, check if all subjects have the same dimensions
    subject_infos = {}
    dimensions = []
    reference_subject_id = None
    reference_subject_info = None
    
    for subject_id in subject_ids:
        subject_info = load_mri_data(subject_id)
        if subject_info:
            subject_infos[subject_id] = subject_info
            dimensions.append(subject_info['original_shape'])
            
            # Use the subject with most common dimensions as reference
            if reference_subject_id is None:
                reference_subject_id = subject_id
                reference_subject_info = subject_info
    
    if not subject_infos:
        print(f"No valid data found for {group_name} group")
        return None
    
    # Count dimension frequencies to find most common
    dim_count = {}
    for dim in dimensions:
        dim_str = str(dim)
        if dim_str in dim_count:
            dim_count[dim_str] += 1
        else:
            dim_count[dim_str] = 1
    
    # Find most common dimension
    most_common_dim_str = max(dim_count.items(), key=lambda x: x[1])[0]
    print(f"Most common dimension in {group_name} group: {most_common_dim_str}")
    
    # Filter subjects to only those with matching dimensions
    matching_subjects = []
    all_data = []
    
    for subject_id, info in subject_infos.items():
        if str(info['original_shape']) == most_common_dim_str:
            matching_subjects.append(subject_id)
            all_data.append(info['data_2d'])
            
            # Update reference subject if needed
            if reference_subject_id not in matching_subjects:
                reference_subject_id = subject_id
                reference_subject_info = info
    
    print(f"Using subjects with matching dimensions: {matching_subjects}")
    
    if not all_data:
        print(f"No subjects with matching dimensions found for {group_name} group")
        return None
    
    # Concatenate all subject data along the time dimension
    combined_data = np.vstack(all_data)
    
    print(f"Combined data shape for {group_name} group: {combined_data.shape}")
    
    # Apply ICA to the combined data
    ica_results = apply_ica(combined_data, n_components)
    
    # Reconstruct component maps using the reference subject's dimensions and affine
    component_maps = reconstruct_component_maps(
        ica_results['component_maps'], 
        reference_subject_info['original_shape']
    )
    
    # Save the group component maps
    group_output_dir = os.path.join(output_dir, f"group_{group_name}")
    os.makedirs(group_output_dir, exist_ok=True)
    
    for i, component_map in enumerate(component_maps):
        # Create a NIfTI image from the component map
        nii_img = nib.Nifti1Image(component_map, reference_subject_info['affine'])
        
        # Save the image
        output_file = os.path.join(group_output_dir, f"component_{i+1}.nii.gz")
        nib.save(nii_img, output_file)
        
        # Create visualization
        fig_dir = os.path.join(group_output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"component_{i+1}.png")
        
        fig = plt.figure(figsize=(10, 6))
        display = plotting.plot_stat_map(
            nii_img, 
            cut_coords=None,  # Let nilearn automatically determine best cut coordinates
            display_mode='ortho',
            title=f"Group {group_name} - Component {i+1}",
            colorbar=True
        )
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {len(component_maps)} group-level component maps for {group_name}")
    
    return {
        'component_maps': component_maps,
        'mixing_matrix': ica_results['mixing_matrix'],
        'ica_model': ica_results['ica_model'],
        'affine': reference_subject_info['affine']
    }

# Function to analyze individual subjects
def analyze_individual_subject(subject_id):
    """Perform ICA analysis on a single subject."""
    print(f"\nAnalyzing subject {subject_id}...")
    
    # Load MRI data
    subject_info = load_mri_data(subject_id)
    
    if not subject_info:
        print(f"Skipping subject {subject_id} due to data loading errors")
        return None
    
    # Apply ICA
    ica_results = apply_ica(subject_info['data_2d'], n_components)
    
    # Reconstruct component maps
    component_maps = reconstruct_component_maps(
        ica_results['component_maps'], 
        subject_info['original_shape']
    )
    
    # Save component maps
    save_component_maps(component_maps, subject_info['affine'], subject_id)
    
    # Visualize component maps
    visualize_component_maps(component_maps, subject_info['affine'], subject_id)
    
    return {
        'component_maps': component_maps,
        'mixing_matrix': ica_results['mixing_matrix'],
        'ica_model': ica_results['ica_model']
    }

# Function to compare components across groups
def compare_group_components(group_results):
    """Compare ICA components across different groups."""
    print("\nComparing group-level components...")
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "group_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create a summary table of groups and their top components
    summary_data = []
    
    for group_name, results in group_results.items():
        if results:
            # Calculate component power (variance explained)
            component_power = np.var(results['mixing_matrix'], axis=0)
            
            # Sort components by power
            sorted_indices = np.argsort(-component_power)
            
            # Add top 5 components to summary
            for rank, idx in enumerate(sorted_indices[:5]):
                summary_data.append({
                    'Group': group_name,
                    'Rank': rank + 1,
                    'Component': idx + 1,
                    'Power': component_power[idx]
                })
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(comparison_dir, "top_components_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"Saved group comparison summary to {summary_csv}")
    
    # Create visualizations comparing the top components across groups
    fig = plt.figure(figsize=(12, 8))
    plt.bar(
        summary_df.index, 
        summary_df['Power'],
        color=[plt.cm.tab10(i) for i in summary_df['Group'].factorize()[0]]
    )
    plt.xticks(
        summary_df.index,
        [f"{row['Group']}-{row['Rank']}" for _, row in summary_df.iterrows()],
        rotation=90
    )
    plt.ylabel('Component Power (Variance)')
    plt.title('Top ICA Components by Group')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "top_components_power.png"), dpi=150)
    plt.close(fig)

# Main execution
def main():
    """Main execution function."""
    print("Starting ICA analysis of brain networks from MRI data...")
    
    # Analyze individual subjects
    individual_results = {}
    all_subject_ids = []
    for group, subject_ids in groups.items():
        all_subject_ids.extend(subject_ids)
    
    for subject_id in all_subject_ids:
        individual_results[subject_id] = analyze_individual_subject(subject_id)
    
    # Extract group-level components
    group_results = {}
    for group_name, subject_ids in groups.items():
        group_results[group_name] = extract_group_components(group_name, subject_ids, n_components)
    
    # Compare components across groups
    compare_group_components(group_results)
    
    print("\nICA analysis complete. Results saved in the ICA_Results directory.")

if __name__ == "__main__":
    main()