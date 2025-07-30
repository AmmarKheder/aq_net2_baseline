#!/bin/bash
JOB_ID=${1:-12037510}
echo "Monitoring job $JOB_ID..."

# Wait for job to start
while true; do
    STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "Job completed or cancelled"
        break
    elif [ "$STATUS" == "RUNNING" ]; then
        echo "Job is running!"
        tail -f logs/rossice_${JOB_ID}.out
        break
    else
        echo "Status: $STATUS - waiting..."
        sleep 10
    fi
done
