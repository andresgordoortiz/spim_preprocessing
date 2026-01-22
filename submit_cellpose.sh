#!/bin/bash
#SBATCH --job-name=cellpose_seg
#SBATCH --output=logs/cellpose_%A_%a.out
#SBATCH --error=logs/cellpose_%A_%a.err
#SBATCH --partition=g                    # GPU Partition
#SBATCH --gres=gpu:1                     # 1 GPU per task
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
OUTPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/results/cellpose_segmentation"

# Container version (same as preprocessing)
CONTAINER_IMAGE="docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-8720eea"

# ==========================================
# CELLPOSE PARAMETERS
# ==========================================
# Model selection: 'cyto', 'cyto2', 'nuclei', or path to custom model
MODEL="nuclei"

# Diameter of cells in pixels (0 = auto-estimate)
DIAMETER=0

# Flow threshold (higher = more conservative segmentation)
FLOW_THRESHOLD=0.4

# Cell probability threshold (higher = fewer cells detected)
CELLPROB_THRESHOLD=0.0

# Channels to use [cytoplasm, nucleus] - [0,0] = grayscale
CHAN="0"
CHAN2="0"

# Additional flags
USE_GPU=true
SAVE_OUTLINES=true
SAVE_FLOWS=true
SAVE_NPY=true  # NPY files saved by default; set false to disable

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
# FILE ARRAY SETUP
# ==========================================
# Get list of all TIFF files in input directory
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.tif" -o -name "*.tiff" \) | sort)

# Get total number of files
TOTAL_FILES=${#FILES[@]}

echo "Total files found: $TOTAL_FILES"

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