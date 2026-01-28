#!/usr/bin/env bash
#SBATCH --no-requeue
#SBATCH --mem 6G
#SBATCH -p c
#SBATCH --qos c_medium
#SBATCH --time 2-00:00:00

# Configure bash
set -e          # exit immediately on error
set -u          # exit immediately if using undefined variables
set -o pipefail # ensure bash pipelines return non-zero status if any of their command fails

# Setup trap function to be run when canceling the pipeline job. It will propagate the SIGTERM signal
# to Nextflow so that all jobs launched by the pipeline will be cancelled too.
_term() {
    echo "Caught SIGTERM signal!"
    # Send SIGTERM to the entire process group to ensure Tower gets the signal
    kill -s SIGTERM -$pid 2>/dev/null || kill -s SIGTERM $pid
    wait $pid
}
trap _term TERM INT


# Load modules - using specific versions for reproducibility
module load build-env/f2022
module load nextflow/25.04.7
module load java/21

# Export Seqera Platform (Tower) token
export TOWER_ACCESS_TOKEN="eyJ0aWQiOiAxMzM2Nn0uZWRlMTAxYmIzZGE4ZDNjMzJjM2M1MmZkZThhNjBhZGI2M2EyNjE4Mg=="

export NXF_OPTS="-Xss4M"

# Check if we're running in a SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running as SLURM job ID: $SLURM_JOB_ID"
    echo "SLURM_CONF is set to: $SLURM_CONF"
else
    echo "Warning: Not running as a SLURM job. Process distribution may be limited."
fi

# Ensure SLURM_CONF is available to child processes
export SLURM_CONF=${SLURM_CONF:-/etc/slurm/slurm.conf}
echo "Using SLURM configuration: $SLURM_CONF"

# Make absolutely sure Nextflow sees this is a SLURM environment
export NXF_EXECUTOR=slurm
export NXF_CLUSTER_SEED=531684

# Ensure all relevant SLURM environment variables are preserved and passed to Nextflow
export SLURM_EXPORT_ENV=ALL

# Preserve additional SLURM variables that Tower might need
export SLURM_JOB_ID SLURM_JOB_NAME SLURM_CLUSTER_NAME

# limit the RAM that can be used by nextflow
export NXF_JVM_ARGS="-Xms2g -Xmx5g -Dexecutor.name=slurm"

# Debug output
echo "Environment variables for Nextflow:"
env | grep -E 'SLURM|NXF'

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Input/Output directories (MODIFY THESE)
INPUT_DIR="./data"
OUTPUT_DIR="./spim_pipeline_output"
CONFIG_JSON="./config_medaka.json"
CHANNEL=1  # Name of the channel to process

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
    --channel "$CHANNEL" \
    -profile "$PROFILE" \
    $RESUME \
    -with-report "$OUTPUT_DIR/reports/nextflow_report_${TIMESTAMP}.html" \
    -with-timeline "$OUTPUT_DIR/reports/nextflow_timeline_${TIMESTAMP}.html" \
    -with-trace "$OUTPUT_DIR/reports/nextflow_trace_${TIMESTAMP}.txt" \
    -with-dag "$OUTPUT_DIR/reports/nextflow_dag_${TIMESTAMP}.html" \
    -with-tower \
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