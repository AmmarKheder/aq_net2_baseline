#!/bin/bash
# Script de récupération automatique en cas d'échec

MAIN_JOB_ID="12050281"
MAX_RETRIES=3
RETRY_COUNT=0

echo "🔧 AUTO-RECOVERY ROSSICE ACTIVÉ"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Attendre que le job se termine
    while squeue -j $MAIN_JOB_ID 2>/dev/null | grep -q $MAIN_JOB_ID; do
        sleep 300  # 5 minutes
    done
    
    echo "Job $MAIN_JOB_ID terminé. Analyse..."
    
    # Vérifier si c'était un succès
    LOG_FILE="/scratch/project_462000640/ammar/rossice/logs/rossice_pm25_${MAIN_JOB_ID}.out"
    ERR_FILE="/scratch/project_462000640/ammar/rossice/logs/rossice_pm25_${MAIN_JOB_ID}.err"
    
    SUCCESS=false
    
    # Critères de succès
    if [ -f "$LOG_FILE" ]; then
        if grep -q "Training completed\|Job terminé\|successfully" "$LOG_FILE"; then
            if ls /scratch/project_462000640/ammar/rossice/checkpoints/rossice_*.ckpt 2>/dev/null; then
                SUCCESS=true
                echo "✅ Entraînement terminé avec succès!"
            fi
        fi
    fi
    
    if [ "$SUCCESS" = true ]; then
        break
    else
        echo "❌ Échec détecté. Analyse de l'erreur..."
        
        # Identifier le problème
        PROBLEM=""
        if [ -f "$ERR_FILE" ] && grep -q "out of memory\|OOM" "$ERR_FILE"; then
            PROBLEM="memory"
        elif [ -f "$ERR_FILE" ] && grep -q "CUDA.*error\|GPU.*error" "$ERR_FILE"; then
            PROBLEM="gpu"
        elif [ -f "$LOG_FILE" ] && grep -q "ModuleNotFoundError\|ImportError" "$LOG_FILE"; then
            PROBLEM="import"
        else
            PROBLEM="unknown"
        fi
        
        echo "Problème identifié: $PROBLEM"
        
        # Créer un script de relance adapté
        RETRY_SCRIPT="/scratch/project_462000640/ammar/rossice/scripts/retry_${RETRY_COUNT}.sh"
        cp /scratch/project_462000640/ammar/rossice/scripts/submit_rossice_fixed.sh "$RETRY_SCRIPT"
        
        # Adapter selon le problème
        case $PROBLEM in
            memory)
                echo "Réduction du batch size..."
                # Ajouter une ligne pour réduire le batch size dans le script
                echo "export ROSSICE_BATCH_SIZE=8" >> "$RETRY_SCRIPT"
                ;;
            gpu)
                echo "Réduction à 1 GPU..."
                sed -i 's/--gres=gpu:mi250:2/--gres=gpu:mi250:1/g' "$RETRY_SCRIPT"
                sed -i 's/WORLD_SIZE=2/WORLD_SIZE=1/g' "$RETRY_SCRIPT"
                ;;
            *)
                echo "Relance standard..."
                ;;
        esac
        
        # Relancer
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Tentative $RETRY_COUNT/$MAX_RETRIES..."
        NEW_JOB=$(sbatch "$RETRY_SCRIPT" | awk '{print $4}')
        MAIN_JOB_ID=$NEW_JOB
        echo "Nouveau job lancé: $MAIN_JOB_ID"
        
        sleep 60
    fi
done

if [ "$SUCCESS" != true ]; then
    echo "❌ ÉCHEC après $MAX_RETRIES tentatives!"
    echo "Intervention manuelle requise."
else
    echo "✅ SUCCÈS!"
    # Lancer l'analyse des résultats
    python3 /scratch/project_462000640/ammar/rossice/auto_report_generator.py
    python3 /scratch/project_462000640/ammar/rossice/visualize_results.py
fi
