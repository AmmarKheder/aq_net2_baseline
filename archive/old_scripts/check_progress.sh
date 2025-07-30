#!/bin/bash
echo "📊 ÉTAT ACTUEL ROSSICE PM2.5"
echo "=========================="
echo "Date: $(date)"
echo ""

# Job status
echo "🚀 JOB:"
squeue -j 12050281 -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %N" 2>/dev/null || echo "Job terminé"
echo ""

# Dernières lignes du log
LOG_FILE="/scratch/project_462000640/ammar/rossice/logs/rossice_pm25_12050281.out"
if [ -f "$LOG_FILE" ]; then
    echo "📋 ACTIVITÉ RÉCENTE:"
    tail -20 "$LOG_FILE" | grep -E "epoch|Epoch|loss|GPU|Training|mae|rmse|Starting|Error|error" || tail -10 "$LOG_FILE"
    echo ""
fi

# Erreurs
ERR_FILE="/scratch/project_462000640/ammar/rossice/logs/rossice_pm25_12050281.err"
if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
    echo "⚠️  ERREURS:"
    tail -10 "$ERR_FILE"
    echo ""
fi

# Checkpoints
echo "💾 CHECKPOINTS:"
ls -lah /scratch/project_462000640/ammar/rossice/checkpoints/rossice_*.ckpt 2>/dev/null | tail -3 || echo "Aucun checkpoint encore"
echo ""

# Monitoring actif
echo "🤖 PROCESSUS DE MONITORING:"
ps aux | grep -E "monitor|automation" | grep -v grep | wc -l | xargs -I {} echo "{} processus actifs"
