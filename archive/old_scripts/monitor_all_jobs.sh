#!/bin/bash
# Monitoring de tous les jobs ROSSICE

echo "🔍 MONITORING COMPLET ROSSICE"
echo "============================"
echo "Date: $(date)"
echo ""

# Jobs actifs
echo "📊 JOBS ACTIFS:"
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %N" | grep -E "rossice|climax"
echo ""

# Dernier job lancé
LATEST_JOB=$(squeue -u $USER -h -o "%i" | grep -E "1205" | tail -1)
if [ -n "$LATEST_JOB" ]; then
    echo "🚀 FOCUS SUR JOB: $LATEST_JOB"
    
    LOG="/scratch/project_462000640/ammar/rossice/logs/*_${LATEST_JOB}.out"
    ERR="/scratch/project_462000640/ammar/rossice/logs/*_${LATEST_JOB}.err"
    
    # Progression
    if ls $LOG 2>/dev/null; then
        echo "📋 PROGRESSION:"
        tail -20 $(ls -t $LOG | head -1) | grep -E "epoch|Epoch|loss|GPU|mae|rmse|%|/|step" || echo "En cours d'initialisation..."
    fi
    
    # Erreurs
    if ls $ERR 2>/dev/null && [ -s $(ls -t $ERR | head -1) ]; then
        echo ""
        echo "⚠️  DERNIÈRES ERREURS:"
        tail -10 $(ls -t $ERR | head -1) | grep -v "sticky modules\|There are messages"
    fi
fi

echo ""
echo "💾 CHECKPOINTS:"
ls -lah /scratch/project_462000640/ammar/rossice/checkpoints/*.ckpt 2>/dev/null | tail -3 || echo "Aucun checkpoint encore"

echo ""
echo "🤖 PROCESSUS ACTIFS:"
ps aux | grep -E "rossice|monitor|automation" | grep -v grep | wc -l
