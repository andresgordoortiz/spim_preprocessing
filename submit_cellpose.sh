#!/bin/bash
#SBATCH --job-name=cellpose_seg
#SBATCH --output=logs/cellpose_%A_%a.out
#SBATCH --error=logs/cellpose_%A_%a.err
#SBATCH --partition=g                    # GPU Partition
#SBATCH --gres=gpu:1                     # 1 GPU per task
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G #Increase dramatically in case you need the NPY or flow files
#SBATCH --time=02:00:00
#SBATCH --exclude=clip-g1-[0-6]          # Exclude P100 nodes (use V100/RTX/A100 only)
#SBATCH --array=0                   # Adjust based on number of files

# ==========================================
# SETUP AND VALIDATION
# ==========================================
set -euo pipefail  # Exit on error, undefined variables, pipe failures

# ==========================================
# CREATE ISOLATED TEMPORARY CACHE
# ==========================================
# Create unique temporary directory for this job instance
JOB_TEMP_DIR="/tmp/cellpose_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$$"
mkdir -p "$JOB_TEMP_DIR"

# Only isolate cache and lock files, NOT the environment itself
# The microscopy_env exists in the container and should not be redirected
export CONDA_PKGS_DIRS="${JOB_TEMP_DIR}/pkgs"
export MAMBA_PKGS_DIRS="${JOB_TEMP_DIR}/pkgs"
export MAMBA_ROOT_PREFIX="${JOB_TEMP_DIR}/mamba"

# Prevent any caching or shared state
export CONDA_NO_PLUGINS=true

# Create required directories
mkdir -p "$CONDA_PKGS_DIRS"
mkdir -p "$MAMBA_ROOT_PREFIX"

echo "=========================================="
echo "Isolated Cache Setup"
echo "Temporary directory: $JOB_TEMP_DIR"
echo "CONDA_PKGS_DIRS: $CONDA_PKGS_DIRS"
echo "MAMBA_ROOT_PREFIX: $MAMBA_ROOT_PREFIX"
echo "=========================================="

# ==========================================
# PARSE COMMAND LINE ARGUMENTS
# ==========================================
# Default values
INPUT_DIR=""
OUTPUT_SUBFOLDER="cellpose_segmentation"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -I|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -O|--output)
            OUTPUT_SUBFOLDER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch $0 -I <input_dir> [-O <output_subfolder_name>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "ERROR: Input directory is required"
    echo "Usage: sbatch $0 -I <input_dir> [-O <output_subfolder_name>]"
    echo "Example: sbatch $0 -I /path/to/images"
    echo "         (creates output in /path/to/images/cellpose_segmentation)"
    exit 1
fi

# Convert to absolute path (required for Singularity bind mounts)
INPUT_DIR=$(realpath "$INPUT_DIR")

# Set output directory as subfolder of input
OUTPUT_DIR="${INPUT_DIR}/${OUTPUT_SUBFOLDER}"

# Create logs directory (must exist before job submission for SLURM)
# Run this BEFORE submitting: mkdir -p logs
if [ ! -d "logs" ]; then
    echo "ERROR: logs directory does not exist. Create it before submitting."
    exit 1
fi

# ==========================================
# USER VARIABLES
# ==========================================

# Container version (same as preprocessing)
CONTAINER_IMAGE="docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-8720eea"

# ==========================================
# CELLPOSE PARAMETERS
# ==========================================
# Model selection: 'cyto', 'cyto2', 'nuclei', or path to custom model
MODEL="/groups/pinheiro/user/guilherme.ventura/for_analysis/SPIM/things_from_Thomas/cellpose/models/cpsam_Gui_tracking_20250801"
MODEL=$(realpath "$MODEL")

# Diameter of cells in pixels (0 = auto-estimate)
# NOT ZERO FOR 3D IMAGES!!!!
DIAMETER=27 #For Medaka

# Flow threshold (higher = more conservative segmentation)
FLOW_THRESHOLD=0.8

# Cell probability threshold (higher = fewer cells detected)
CELLPROB_THRESHOLD=0.0

# Additional flags
USE_GPU=true
DO_3D=true  # Set true for 3D images
SAVE_TIF=true  # MUST use TIF for 3D (PNG doesn't work with 3D)
SAVE_FLOWS=false
SAVE_NPY=false  # NPY files saved by default

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

# Get base filename without extension (for output naming)
BASE_NAME=$(basename "$FILENAME" .tif)
BASE_NAME=$(basename "$BASE_NAME" .tiff)

echo "Processing file: $FILENAME"
echo "Base name: $BASE_NAME"
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
echo "  Use GPU: $USE_GPU"
echo "  3D Mode: $DO_3D"
echo "  Save TIF: $SAVE_TIF"
echo "  Save flows: $SAVE_FLOWS"
echo "  Save NPY masks: $SAVE_NPY"
echo "=========================================="

# ==========================================
# CREATE PARAMETER LOG FILE
# ==========================================
PARAM_LOG="${OUTPUT_DIR}/${BASE_NAME}_parameters.txt"

echo "Creating parameter log: $PARAM_LOG"

cat > "$PARAM_LOG" << EOF
Cellpose Segmentation Parameters
=================================
Processed: $(date)
Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
Node: $SLURM_NODELIST

Input File
----------
Filename: $FILENAME
Base name: $BASE_NAME
Full path: $INPUT_FILE

Output Files
------------
Output directory: $OUTPUT_DIR
Mask file: ${BASE_NAME}_Cellseg.tif
Parameter log: ${BASE_NAME}_parameters.txt

Cellpose Parameters
-------------------
Model: $MODEL
Diameter: $DIAMETER pixels
Flow threshold: $FLOW_THRESHOLD
Cell probability threshold: $CELLPROB_THRESHOLD
Use GPU: $USE_GPU
3D Mode: $DO_3D
Save TIF masks: $SAVE_TIF
Save flows: $SAVE_FLOWS
Save NPY masks: $SAVE_NPY

Container
---------
Image: $CONTAINER_IMAGE

System Information
------------------
Hostname: $(hostname)
CPUs per task: $SLURM_CPUS_PER_TASK
Memory: 64G
Time limit: 02:00:00
EOF

# Append GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "" >> "$PARAM_LOG"
    echo "GPU Information" >> "$PARAM_LOG"
    echo "---------------" >> "$PARAM_LOG"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> "$PARAM_LOG" 2>&1
fi

echo "Parameter log created successfully"
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
# Use --image_path for single file and --savedir for output location
CELLPOSE_CMD="cellpose --image_path $INPUT_FILE --savedir $OUTPUT_DIR"
CELLPOSE_CMD+=" --pretrained_model $MODEL"
CELLPOSE_CMD+=" --diameter $DIAMETER"
CELLPOSE_CMD+=" --flow_threshold $FLOW_THRESHOLD"
CELLPOSE_CMD+=" --cellprob_threshold $CELLPROB_THRESHOLD"

if [ "$USE_GPU" = true ]; then
    CELLPOSE_CMD+=" --use_gpu"
fi

if [ "$DO_3D" = true ]; then
    CELLPOSE_CMD+=" --do_3D"
fi

# CRITICAL: For 3D images, MUST use --save_tif, NOT --save_png
if [ "$SAVE_TIF" = true ]; then
    CELLPOSE_CMD+=" --save_tif"
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

# Append command to parameter log
echo "" >> "$PARAM_LOG"
echo "Command Executed" >> "$PARAM_LOG"
echo "----------------" >> "$PARAM_LOG"
echo "$CELLPOSE_CMD" >> "$PARAM_LOG"

# ==========================================
# SINGULARITY EXECUTION WITH ISOLATION
# ==========================================
echo "Starting Cellpose segmentation..."

# Start timing
START_TIME=$SECONDS

# Bind input directory, output directory, model directory, AND temporary directory
MODEL_DIR=$(dirname "$MODEL")

# Export cache isolation variables to container
# Use APPTAINERENV_ prefix (modern Singularity/Apptainer)
export APPTAINERENV_CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
export APPTAINERENV_MAMBA_PKGS_DIRS="$MAMBA_PKGS_DIRS"
export APPTAINERENV_CONDA_NO_PLUGINS="true"
export APPTAINERENV_PYTHONDONTWRITEBYTECODE="1"
export APPTAINERENV_PYTHONUNBUFFERED="1"

singularity exec --nv \
    --cleanenv \
    --contain \
    -B "$INPUT_DIR" \
    -B "$OUTPUT_DIR" \
    -B "$MODEL_DIR" \
    -B "$JOB_TEMP_DIR" \
    "$CONTAINER_IMAGE" \
    /bin/bash -c "
        # Set cache isolation variables inside container
        # Only redirect package cache, not environment lookup
        export CONDA_PKGS_DIRS='$CONDA_PKGS_DIRS'
        export MAMBA_PKGS_DIRS='$MAMBA_PKGS_DIRS'
        export CONDA_NO_PLUGINS=true
        export PYTHONDONTWRITEBYTECODE=1
        export PYTHONUNBUFFERED=1

        # Run micromamba with the existing container environment
        /bin/micromamba run -n microscopy_env $CELLPOSE_CMD
    "

EXIT_CODE=$?

END_TIME=$SECONDS
ELAPSED_TIME=$((END_TIME - START_TIME))

# ==========================================
# RENAME OUTPUT TO MATCH IMAGEJ CONVENTION
# ==========================================
if [ $EXIT_CODE -eq 0 ]; then
    # Cellpose default output: {basename}_cp_masks.tif
    CELLPOSE_OUTPUT="${OUTPUT_DIR}/${BASE_NAME}_cp_masks.tif"

    # NEW: Include diameter in the final filename
    IMAGEJ_STYLE_OUTPUT="${OUTPUT_DIR}/${BASE_NAME}_diam${DIAMETER}_Cellseg.tif"

    if [ -f "$CELLPOSE_OUTPUT" ]; then
        echo "Renaming output to include diameter..."
        mv "$CELLPOSE_OUTPUT" "$IMAGEJ_STYLE_OUTPUT"
        echo "Renamed: $CELLPOSE_OUTPUT -> $IMAGEJ_STYLE_OUTPUT"
    else
        echo "WARNING: Expected output file not found: $CELLPOSE_OUTPUT"
        # Check if maybe it's already named correctly or listing contents
        ls -lh "$OUTPUT_DIR"
    fi
fi

# ==========================================
# UPDATE PARAMETER LOG WITH RESULTS
# ==========================================
echo "" >> "$PARAM_LOG"
echo "Execution Results" >> "$PARAM_LOG"
echo "-----------------" >> "$PARAM_LOG"
echo "Exit code: $EXIT_CODE" >> "$PARAM_LOG"
echo "Status: $([ $EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" >> "$PARAM_LOG"
echo "Elapsed time: ${ELAPSED_TIME} seconds" >> "$PARAM_LOG"
echo "End time: $(date)" >> "$PARAM_LOG"

if [ $EXIT_CODE -eq 0 ]; then
    echo "" >> "$PARAM_LOG"
    echo "Output Files Generated" >> "$PARAM_LOG"
    echo "----------------------" >> "$PARAM_LOG"
    ls -lh "$OUTPUT_DIR"/${BASE_NAME}* 2>/dev/null | awk '{print $9, "(" $5 ")"}' >> "$PARAM_LOG" || echo "No output files found" >> "$PARAM_LOG"
fi

# ==========================================
# CLEANUP ISOLATED ENVIRONMENT
# ==========================================
echo "=========================================="
echo "Cleaning up isolated temporary directory: $JOB_TEMP_DIR"
rm -rf "$JOB_TEMP_DIR"
echo "Cleanup complete"

# ==========================================
# COMPLETION
# ==========================================
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Segmentation completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"
    echo "Parameter log: $PARAM_LOG"

    # List output files for this image
    echo "Output files for $BASE_NAME:"
    ls -lh "$OUTPUT_DIR"/${BASE_NAME}* 2>/dev/null || echo "  No output files found"
else
    echo "ERROR: Segmentation failed with exit code $EXIT_CODE"
    echo "Check parameter log for details: $PARAM_LOG"
fi

echo "End time: $(date)"
echo "Elapsed time: ${ELAPSED_TIME} seconds"
echo "=========================================="

exit $EXIT_CODE