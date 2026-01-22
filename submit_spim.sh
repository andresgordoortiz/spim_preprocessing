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
# VARIABLES DEL USUARIO
# ==========================================

# Define the specific file you want to process
INPUT_FILE="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/data/input_low-res/YOUR_IMAGE_NAME.tif"
OUTPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/results/single_run"
PSF_FILE="/groups/pinheiro/user/guilherme.ventura/for_analysis/SPIM/things_from_Andres/scripts_from_Andres/image_processing_script/psf_models/PSF_small_low-res.tif"
SCRIPT_PATH="spim_pipeline.py"

# Extraction of path components for mounting
INPUT_DIR=$(dirname "$INPUT_FILE")

# Parameters
PARAMS=(
    "--image_scaling 0.5"
    "--niter 3"
    "--niterz 3"
    "--percentile_low 40"
    "--percentile_high 99.99"
    "--sigma 1.0"
)

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Procesando archivo: $INPUT_FILE"
echo "=========================================="

# ==========================================
# EJECUCIÓN CON SINGULARITY
# ==========================================

singularity exec --nv \
    -B "$INPUT_DIR" \
    -B "$OUTPUT_DIR" \
    -B "$(dirname "$PSF_FILE")" \
    -B "$(pwd)" \
    ghcr.io/andresgordoortiz/spim_preprocessing:sha-b5d9ab5 \
    python "$SCRIPT_PATH" \
    --input_file "$INPUT_FILE" \
    --outdir "$OUTPUT_DIR" \
    --psf_path "$PSF_FILE" \
    "${PARAMS[@]}"

echo "Procesamiento finalizado."