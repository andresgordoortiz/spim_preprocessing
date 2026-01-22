#!/bin/bash
#SBATCH --job-name=spim_process
#SBATCH --output=logs/spim_%A_%a.out
#SBATCH --error=logs/spim_%A_%a.err
#SBATCH --partition=g                    # OBLIGATORIO: Partición con GPU (RedLionfish requiere CUDA)
#SBATCH --gres=gpu:1                     # Solicitamos 1 GPU por tarea
#SBATCH --cpus-per-task=8                # 8 CPUs para operaciones numpy/clahe
#SBATCH --mem=64G                        # 64GB de RAM (Stack 3D son pesados)
#SBATCH --time=02:00:00                  # Límite de tiempo por imagen
#SBATCH --array=1

# ==========================================
# VARIABLES DEL USUARIO (EDITAR AQUI)
# ==========================================

# Rutas en el host (cluster)
INPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/data/input_low-res"
OUTPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/results/first_try_sbatch"
PSF_FILE="/groups/pinheiro/user/guilherme.ventura/for_analysis/SPIM/things_from_Andres/scripts_from_Andres/image_processing_script/psf_models/PSF_small_low-res.tif"
SCRIPT_PATH="spim_pipeline.py"

# Parámetros del pipeline (puedes cambiar valores aquí)
PARAMS=(
    "--image_scaling 0.5"
    "--niter 3"
    "--niterz 3"
    "--percentile_low 40"
    "--percentile_high 99.99"
    "--sigma 1.0"
    # Añadir flags booleanos si se desea desactivar pasos:
    # "--no_clahe"
    # "--no_shading"
)

# Crear directorio de logs si no existe
mkdir -p logs

# ==========================================
# LOGICA DEL ARRAY (NO EDITAR LOGICA BASICA)
# ==========================================


# Obtener lista de imágenes (tif, tiff, nd2)
shopt -s nullglob
FILES=(${INPUT_DIR}/*.tif ${INPUT_DIR}/*.tiff ${INPUT_DIR}/*.nd2)
NUM_FILES=${#FILES[@]}

# Chequear que el ID del array sea válido para la cantidad de archivos
if [ $SLURM_ARRAY_TASK_ID -ge $NUM_FILES ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID excede el número de archivos ($NUM_FILES). Saliendo."
    exit 0
fi

CURRENT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
FILENAME=$(basename "$CURRENT_FILE")

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $HOSTNAME"
echo "Procesando archivo: $FILENAME"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv
echo "=========================================="

# ==========================================
# EJECUCIÓN CON SINGULARITY
# ==========================================

# Construimos el comando.
# Importante: --nv habilita el soporte de GPU en Singularity.
# -B monta las carpetas necesarias para que el contenedor las vea.

singularity exec --nv \
    -B "$INPUT_DIR" \
    -B "$OUTPUT_DIR" \
    -B "$(dirname "$PSF_FILE")" \
    -B "$(dirname "$SCRIPT_PATH")" \
    ghcr.io/andresgordoortiz/spim_preprocessing:sha-b5d9ab5 \
    python "$SCRIPT_PATH" \
    --input_file "$CURRENT_FILE" \
    --outdir "$OUTPUT_DIR" \
    --psf_path "$PSF_FILE" \
    ${PARAMS[@]}

echo "Procesamiento finalizado para $FILENAME"