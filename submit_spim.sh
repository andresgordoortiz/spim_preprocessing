#!/bin/bash
#SBATCH --job-name=spim_single
#SBATCH --output=logs/spim_single_%j.out
#SBATCH --error=logs/spim_single_%j.err
#SBATCH --partition=g                    # GPU Partition
#SBATCH --gres=gpu:1                     # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

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
# IMPORTANT: INPUT_FILE must be a specific file, not a directory
INPUT_FILE="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/data/input_low-res/t0120_Channel_2_low-res.tif"
OUTPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/results/first_try_sbatch"
PSF_FILE="/groups/pinheiro/user/guilherme.ventura/for_analysis/SPIM/things_from_Andres/scripts_from_Andres/image_processing_script/psf_models/PSF_small_low-res.tif"
SCRIPT_PATH="spim_pipeline_fixed.py"

# Container version (update this when you update the container)
CONTAINER_IMAGE="docker://ghcr.io/andresgordoortiz/spim_preprocessing:sha-b5d9ab5"

# ==========================================
# INPUT VALIDATION
# ==========================================
echo "=========================================="
echo "SPIM Preprocessing Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Validate input file exists and is a file
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file does not exist or is not a file: $INPUT_FILE"
    exit 1
fi

# Validate PSF file exists
if [ ! -f "$PSF_FILE" ]; then
    echo "ERROR: PSF file does not exist: $PSF_FILE"
    exit 1
fi

# Validate script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script does not exist: $SCRIPT_PATH"
    exit 1
fi

# Extract directory paths for mounting
INPUT_DIR=$(dirname "$INPUT_FILE")
PSF_DIR=$(dirname "$PSF_FILE")
WORK_DIR=$(pwd)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "PSF file: $PSF_FILE"
echo "Container: $CONTAINER_IMAGE"
echo "=========================================="

# ==========================================
# PROCESSING PARAMETERS
# ==========================================
# Define parameters as separate array elements
PARAMS=(
    --image_scaling 0.5
    --niter 3
    --niterz 3
    --percentile_low 40
    --percentile_high 99.99
    --sigma 1.0
    --min_v 0
    --max_v 65535
    --resolution_px0 10
    --resolution_pz0 10
    --noise_lvl 2
    --padding 32
)

# Log parameters for reproducibility
echo "Processing parameters:"
printf '%s\n' "${PARAMS[@]}"
echo "=========================================="

# ==========================================
# GPU CHECK
# ==========================================
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "WARNING: nvidia-smi not available on login node (expected)"
}
echo "=========================================="

# ==========================================
# REPRODUCIBILITY METADATA
# ==========================================
# Log git information if in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Git Information:"
    echo "  Commit: $(git rev-parse HEAD)"
    echo "  Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "  Status: $(git diff-index --quiet HEAD -- && echo 'clean' || echo 'modified')"
    echo "=========================================="
fi

# ==========================================
# SINGULARITY EXECUTION
# ==========================================
echo "Starting Singularity container execution..."
echo "Mounted directories:"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  PSF: $PSF_DIR"
echo "  Working: $WORK_DIR"
echo "=========================================="

singularity exec --nv \
    -B "$INPUT_DIR" \
    -B "$OUTPUT_DIR" \
    -B "$PSF_DIR" \
    -B "$WORK_DIR" \
    "$CONTAINER_IMAGE" \
    python "$SCRIPT_PATH" \
    --input_file "$INPUT_FILE" \
    --outdir "$OUTPUT_DIR" \
    --psf_path "$PSF_FILE" \
    "${PARAMS[@]}"

EXIT_CODE=$?

# ==========================================
# COMPLETION
# ==========================================
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Processing completed successfully!"
    echo "Output saved to: $OUTPUT_DIR"

    # List output files
    echo "Output files:"
    ls -lh "$OUTPUT_DIR"/*.tif 2>/dev/null || echo "  No output files found"
else
    echo "ERROR: Processing failed with exit code $EXIT_CODE"
fi

echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE