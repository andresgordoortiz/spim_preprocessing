#!/bin/bash
#
# Example submission script for SPIM Pipeline on SLURM
#
# Usage:
#   bash submit_pipeline.sh
#
# Or submit as a job:
#   sbatch submit_pipeline.sh
#

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Input/Output directories (MODIFY THESE)
INPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/data/input_low-res"
OUTPUT_DIR="/groups/pinheiro/user/andres.gordo/projects/spim_preprocessing/results/nextflow_pipeline"
CONFIG_JSON="./config_medaka.json"

# Nextflow parameters
PROFILE="standard"  # Options: standard, highres, local
RESUME="-resume"    # Use -resume to continue from last step, or leave empty

# Pipeline location
PIPELINE_SCRIPT="./spim_pipeline.nf"

# ============================================================================
# VALIDATION
# ============================================================================

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_JSON" ]; then
    echo "ERROR: Configuration file does not exist: $CONFIG_JSON"
    exit 1
fi

# Check if pipeline script exists
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script does not exist: $PIPELINE_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

# Load Java module (if needed on your cluster)
# module load java/11

# Or ensure Nextflow is in PATH
export PATH="$HOME/.nextflow:$PATH"

# Set Nextflow temporary directory
export NXF_TEMP="$OUTPUT_DIR/.nextflow_temp"
mkdir -p "$NXF_TEMP"

# Optional: Set Singularity cache directory
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache"
mkdir -p "$SINGULARITY_CACHEDIR"

# ============================================================================
# LOGGING
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/pipeline_submission_${TIMESTAMP}.log"

echo "============================================================================"
echo "SPIM Pipeline Submission"
echo "============================================================================"
echo "Timestamp       : $TIMESTAMP"
echo "Input directory : $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Configuration   : $CONFIG_JSON"
echo "Profile         : $PROFILE"
echo "Resume          : ${RESUME:-false}"
echo "Log file        : $LOG_FILE"
echo "============================================================================"

# Count input files
N_FILES=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.tif" -o -name "*.tiff" \) | wc -l)
echo "Found $N_FILES TIFF files to process"
echo "============================================================================"
echo ""

# ============================================================================
# EXECUTE PIPELINE
# ============================================================================

# Run Nextflow pipeline
nextflow run "$PIPELINE_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_json "$CONFIG_JSON" \
    -profile "$PROFILE" \
    $RESUME \
    -with-report "$OUTPUT_DIR/reports/nextflow_report_${TIMESTAMP}.html" \
    -with-timeline "$OUTPUT_DIR/reports/nextflow_timeline_${TIMESTAMP}.html" \
    -with-trace "$OUTPUT_DIR/reports/nextflow_trace_${TIMESTAMP}.txt" \
    -with-dag "$OUTPUT_DIR/reports/nextflow_dag_${TIMESTAMP}.html" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully!"
    echo ""
    echo "Results available in:"
    echo "  - Preprocessed: $OUTPUT_DIR/01_preprocessed/"
    echo "  - Segmented   : $OUTPUT_DIR/02_segmented/"
    echo "  - Tracked     : $OUTPUT_DIR/03_tracked/"
    echo "  - Reports     : $OUTPUT_DIR/reports/"
else
    echo "Pipeline failed with exit code: $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi
echo "============================================================================"

exit $EXIT_CODE