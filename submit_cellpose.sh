#!/bin/bash
#SBATCH --job-name=cellpose_seg
#SBATCH --output=logs/cellpose_%A_%a.out
#SBATCH --error=logs/cellpose_%A_%a.err
#SBATCH --partition=g                    # GPU Partition
#SBATCH --gres=gpu:1                     # 1 GPU per task
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=clip-g1-[0-6]          # Exclude P100 nodes (use V100/RTX/A100 only)
#SBATCH --array=0                   # Adjust based on number of files

# ==========================================
# SETUP AND VALIDATION
# ==========================================
set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Create logs directory (must exist before job submission for SLURM)
# Run this BEFORE submitting: mkdir -p logs
if [ ! -d "logs" ]; then
    echo "ERROR: logs directory does not exist. Create it before submitting."
    exit 1
fi

# ==========================================
# USER VARIABLES
# ==========================================
# Input directory containing processed TIFF files
INPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/results/first_try_sbatch"
OUTPUT_DIR="$INPUT_DIR/cellpose_segmentation"

# Container version (same as preprocessing)
CONTAINER_IMAGE="docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-8720eea"

# ==========================================
# CELLPOSE PARAMETERS
# ==========================================
# Model selection: 'cyto', 'cyto2', 'nuclei', or path to custom model
MODEL="cpsam_Gui_tracking_20250801"

# Diameter of cells in pixels (0 = auto-estimate)
# NOT ZERO FOR 3D IMAGES!!!!
DIAMETER=27 #For Medaka

# Flow threshold (higher = more conservative segmentation)
# NOTE: ImageJ macro uses -0.8, but Cellpose treats negative values as positive
FLOW_THRESHOLD=0.8

# Cell probability threshold (higher = fewer cells detected)
CELLPROB_THRESHOLD=0.0

# Channels to use [cytoplasm, nucleus]
# ImageJ macro uses ch1=1, ch2=0 (grayscale on channel 1, no nuclear channel)
CHAN="1"
CHAN2="0"  # Changed from "2" to "0" to match ImageJ macro

# Additional flags
USE_GPU=true
DO_3D=true  # Set true for 3D images
SAVE_OUTLINES=true
SAVE_FLOWS=true
SAVE_NPY=true  # NPY files saved by default; set false to disable

# File exclusion patterns (matching ImageJ macro logic)
EXCLUDE_PATTERNS=("Cellseg.tif" "Sarco.tif" "Label")

# ==========================================
# INPUT VALIDATION
# ==========================================
echo "=========================================="
echo "Cellpose Segmentation Job"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==========================================
# FILE ARRAY SETUP WITH EXCLUSIONS
# ==========================================
# Function to check if file should be excluded
should_exclude() {
    local filename="$1"

    # Exclude .ijm files
    if [[ "$filename" == *.ijm ]]; then
        return 0  # exclude
    fi

    # Check against exclusion patterns
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ "$filename" == *"$pattern"* ]]; then
            return 0  # exclude
        fi
    done

    return 1  # don't exclude
}

# Get list of all TIFF files, excluding unwanted patterns
TEMP_FILES=()
while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    if ! should_exclude "$filename"; then
        TEMP_FILES+=("$file")
    fi
done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.tif" -o -name "*.tiff" \) -print0 | sort -z)

# Convert to regular array
FILES=("${TEMP_FILES[@]}")

# Get total number of files
TOTAL_FILES=${#FILES[@]}

echo "Total files found (after exclusions): $TOTAL_FILES"

# Check if array task ID is valid
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_FILES" ]; then
    echo "WARNING: Array task ID ($SLURM_ARRAY_TASK_ID) exceeds number of files ($TOTAL_FILES)"
    echo "Nothing to process for this task. Exiting gracefully."
    exit 0
fi

# Get the file for this array task
INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
FILENAME=$(basename "$INPUT_FILE")

echo "Processing file: $FILENAME"
echo "Input path: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Container: $CONTAINER_IMAGE"
echo "=========================================="

# ==========================================
# CELLPOSE PARAMETERS LOG
# ==========================================
echo "Cellpose Parameters:"
echo "  Model: $MODEL"
echo "  Diameter: $DIAMETER"
echo "  Flow threshold: $FLOW_THRESHOLD"
echo "  Cell probability threshold: $CELLPROB_THRESHOLD"
echo "  Channels: [$CHAN, $CHAN2]"
echo "  Use GPU: $USE_GPU"
echo "  3D Mode: $DO_3D"
echo "  Save outlines (PNG): $SAVE_OUTLINES"
echo "  Save flows: $SAVE_FLOWS"
echo "  Save NPY masks: $SAVE_NPY"
echo "=========================================="

# ==========================================
# GPU CHECK
# ==========================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "WARNING: nvidia-smi not available"
}
echo "=========================================="

# ==========================================
# BUILD CELLPOSE COMMAND
# ==========================================
CELLPOSE_CMD="cellpose --dir $INPUT_DIR --image_path $INPUT_FILE --savedir $OUTPUT_DIR"
CELLPOSE_CMD+=" --pretrained_model $MODEL"
CELLPOSE_CMD+=" --diameter $DIAMETER"
CELLPOSE_CMD+=" --flow_threshold $FLOW_THRESHOLD"
CELLPOSE_CMD+=" --cellprob_threshold $CELLPROB_THRESHOLD"
CELLPOSE_CMD+=" --chan $CHAN --chan2 $CHAN2"

if [ "$USE_GPU" = true ]; then
    CELLPOSE_CMD+=" --use_gpu"
fi

if [ "$DO_3D" = true ]; then
    CELLPOSE_CMD+=" --do_3D"
fi

if [ "$SAVE_OUTLINES" = true ]; then
    CELLPOSE_CMD+=" --save_png"
fi

if [ "$SAVE_FLOWS" = true ]; then
    CELLPOSE_CMD+=" --save_flows"
fi

if [ "$SAVE_NPY" = false ]; then
    CELLPOSE_CMD+=" --no_npy"
fi

# Add verbose output
CELLPOSE_CMD+=" --verbose"

echo "Cellpose command:"
echo "$CELLPOSE_CMD"
echo "=========================================="

# ==========================================
# SINGULARITY EXECUTION
# ==========================================
echo "Starting Cellpose segmentation..."

singularity exec --nv \
    -B "$INPUT_DIR" \
    -B "$OUTPUT_DIR" \
    "$CONTAINER_IMAGE" \
    /bin/micromamba run -n microscopy_env \
    $CELLPOSE_CMD

EXIT_CODE=$?

# ==========================================
# RENAME OUTPUT TO MATCH IMAGEJ CONVENTION
# ==========================================
if [ $EXIT_CODE -eq 0 ]; then
    # Get base filename without extension
    BASE_NAME=$(basename "$FILENAME" .tif)
    BASE_NAME=$(basename "$BASE_NAME" .tiff)

    # Cellpose default output: {basename}_cp_masks.tif
    # ImageJ output: {basename}_Cellseg.tif
    CELLPOSE_OUTPUT="${OUTPUT_DIR}/${BASE_NAME}_cp_masks.tif"
    IMAGEJ_STYLE_OUTPUT="${OUTPUT_DIR}/${BASE_NAME}_Cellseg.tif"

    if [ -f "$CELLPOSE_OUTPUT" ]; then
        echo "Renaming output to match ImageJ convention..."
        mv "$CELLPOSE_OUTPUT" "$IMAGEJ_STYLE_OUTPUT"
        echo "Renamed: $CELLPOSE_OUTPUT -> $IMAGEJ_STYLE_OUTPUT"
    fi
fi

# ==========================================
# COMPLETION
# ==========================================
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Segmentation completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"

    # List output files for this image
    echo "Output files for $FILENAME:"
    BASE_NAME=$(basename "$FILENAME" .tif)
    BASE_NAME=$(basename "$BASE_NAME" .tiff)
    ls -lh "$OUTPUT_DIR"/${BASE_NAME}* 2>/dev/null || echo "  No output files found"
else
    echo "ERROR: Segmentation failed with exit code $EXIT_CODE"
fi

echo "End time: $(date)"
echo "Elapsed time: $SECONDS seconds"
echo "=========================================="

exit $EXIT_CODE