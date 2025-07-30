#!/bin/bash
JOB_ID=12037492
LOG_FILE="logs/climax_caqra_${JOB_ID}.out"
ERR_FILE="logs/climax_caqra_${JOB_ID}.err"

echo "Monitoring job $JOB_ID..."
echo "Waiting for job to start..."

# Wait for job to start
while true; do
    STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "Job $JOB_ID not found - it may have completed or been cancelled"
        break
    elif [ "$STATUS" == "RUNNING" ] || [ "$STATUS" == "R" ]; then
        echo "Job is now running!"
        break
    else
        echo "Job status: $STATUS - waiting..."
        sleep 10
    fi
done

# Check if log files exist
if [ -f "$LOG_FILE" ]; then
    echo "Log file found at: $LOG_FILE"
    echo "Starting log monitoring (Ctrl+C to stop)..."
    tail -f "$LOG_FILE"
else
    echo "Log file not found at: $LOG_FILE"
    echo "Checking for any job output files..."
    ls -la logs/climax_caqra_${JOB_ID}* 2>/dev/null || echo "No log files found yet"
fi
