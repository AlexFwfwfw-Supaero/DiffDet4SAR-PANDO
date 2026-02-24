#!/bin/bash
###############################################################################
# Submit train.slurm and monitor until completion
#
# Usage (on PANDO frontend):
#   cd ~/DiffDet4SAR-project/DiffDet4SAR
#   bash pando/submit_and_monitor.sh
###############################################################################

USER="a.jesus"
SLURM_FILE="pando/train.slurm"
INTERVAL=10  # seconds between squeue checks

# Submit the job
echo "Submitting $SLURM_FILE..."
OUTPUT=$(sbatch "$SLURM_FILE")
echo "$OUTPUT"

# Extract job ID
JOB_ID=$(echo "$OUTPUT" | grep -oP '(?<=Submitted batch job )\d+')
if [ -z "$JOB_ID" ]; then
    echo "ERROR: Could not parse job ID. Exiting."
    exit 1
fi

echo ""
echo "Job ID:   $JOB_ID"
echo "Log file: diffdet4sar_${JOB_ID}.out"
echo ""
echo "Monitoring (Ctrl+C to stop watching, job keeps running)..."
echo "--------------------------------------------------------------"

# Monitor loop
while true; do
    STATUS=$(squeue -u "$USER" -j "$JOB_ID" --format="%.8i %.9P %.8T %.10M %.6D %R" --noheader 2>/dev/null)

    if [ -z "$STATUS" ]; then
        echo "[$(date +%H:%M:%S)] Job $JOB_ID no longer in queue."
        echo ""
        # Show final status via sacct if available
        if command -v sacct &> /dev/null; then
            echo "Final status:"
            sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed --noheader
        fi
        echo ""
        echo "Check logs: cat diffdet4sar_${JOB_ID}.out"
        break
    fi

    echo "[$(date +%H:%M:%S)] $STATUS"
    sleep "$INTERVAL"
done
