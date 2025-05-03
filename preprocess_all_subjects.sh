#!/bin/bash

# ======= CONFIGURATION =======
FSLDIR="/Users/linikagoel/fsl"
export PATH="$FSLDIR/bin:$PATH"
TR=1.0

# ======= LOOP OVER SUBJECTS =======
BAD_SUBJECTS=()

for i in $(seq -w 1 25); do
    SUBJ="sub-${i}"
    FUNC_RAW="/Users/linikagoel/Documents/NeuroImage_Processing/Final_Project/${SUBJ}/func/${SUBJ}_task-rest_run-01_bold.nii"
    PREPROC_DIR="/Users/linikagoel/Documents/NeuroImage_Processing/Final_Project/${SUBJ}/func/preprocessed"

    echo "🔁 Processing $SUBJ ..."

    # Check if file exists
    if [ ! -f "$FUNC_RAW" ]; then
        echo "❌ $FUNC_RAW not found. Skipping $SUBJ."
        BAD_SUBJECTS+=("$SUBJ")
        continue
    fi

    # Check if image is readable
    fslslice "$FUNC_RAW" temp_slice.nii > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ $FUNC_RAW appears corrupted. Skipping $SUBJ."
        BAD_SUBJECTS+=("$SUBJ")
        continue
    fi
    rm -f temp_slice.nii

    # Delete and recreate output folder
    rm -rf "$PREPROC_DIR"
    mkdir -p "$PREPROC_DIR" || exit 1
    cd "$PREPROC_DIR" || exit 1

    # ======= CHECK REQUIRED TOOLS =======
    REQUIRED_TOOLS=("mcflirt" "fslmaths" "gunzip")
    for tool in "${REQUIRED_TOOLS[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            echo "❌ ERROR: Required FSL tool '$tool' not found."
            exit 1
        fi
    done

    # ======= STEP 1: Motion Correction =======
    echo "🔄 Motion correcting..."
    mcflirt -in "$FUNC_RAW" -out mc_func -plots || exit 1

    # ======= STEP 2: Mean Functional Image =======
    echo "🧠 Computing mean functional image..."
    fslmaths mc_func -Tmean mean_func || exit 1

    # ======= STEP 3: Mask Functional =======
    echo "🧠 Masking functional with mean_func..."
    fslmaths mc_func -mas mean_func masked_func || exit 1

    # ======= STEP 4: Spatial Smoothing =======
    echo "💨 Spatial smoothing..."
    fslmaths masked_func -s 1.5 smoothed_func || exit 1

    # ======= STEP 5: Temporal Filtering =======
    echo "⏳ Temporal filtering (0.1 Hz)..."
    LP_SIGMA=$(echo "0.5 * $TR / 0.1" | bc -l)
    fslmaths smoothed_func -bptf $LP_SIGMA -1 filtered_func || exit 1

    # ======= STEP 6: Unzip for SPM =======
    echo "📂 Unzipping filtered_func and mean_func for SPM..."
    gunzip -f filtered_func.nii.gz mean_func.nii.gz || exit 1

    echo "✅ DONE with $SUBJ"
    echo "--------------------------------------------"
done

# ======= SUMMARY =======
if [ ${#BAD_SUBJECTS[@]} -gt 0 ]; then
    echo "⚠️ The following subjects were skipped due to missing or corrupted files:"
    for subj in "${BAD_SUBJECTS[@]}"; do
        echo "   - $subj"
    done
else
    echo "🎉 All subjects processed successfully!"
fi

