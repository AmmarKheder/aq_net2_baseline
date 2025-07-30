#!/bin/bash
# Script de monitoring automatique pour ROSSICE PM2.5

LOG_DIR="/scratch/project_462000640/ammar/rossice/logs"
CHECKPOINT_DIR="/scratch/project_462000640/ammar/rossice/checkpoints"
RESULTS_DIR="/scratch/project_462000640/ammar/rossice/results"
JOB_ID="12049899"

# Créer les dossiers nécessaires
mkdir -p $LOG_DIR $CHECKPOINT_DIR $RESULTS_DIR

echo "🤖 ROSSICE AUTO-MONITOR ACTIVÉ"
echo "📅 Démarré le: $(date)"
echo "🎯 Job ID: $JOB_ID"
echo "=" | tr = '='*60

# Fonction pour vérifier le statut du job
check_job_status() {
    squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID
    return $?
}

# Fonction pour extraire les métriques du log
extract_metrics() {
    local log_file="$LOG_DIR/climax_caqra_${JOB_ID}.out"
    if [ -f "$log_file" ]; then
        echo "📊 MÉTRIQUES ACTUELLES:"
        # Chercher la dernière epoch
        grep -E "Epoch|train_loss|val_loss|mae|rmse" "$log_file" | tail -20
        echo ""
        # Chercher les erreurs GPU
        grep -i "error\|fail\|cuda\|gpu" "$log_file" | tail -5
    fi
}

# Fonction pour vérifier les checkpoints
check_checkpoints() {
    echo "💾 CHECKPOINTS:"
    ls -lah $CHECKPOINT_DIR/rossice_* 2>/dev/null | tail -5 || echo "Aucun checkpoint trouvé encore"
    echo ""
}

# Fonction pour analyser les prédictions
analyze_predictions() {
    local pred_file="$RESULTS_DIR/pm25_predictions_*.npz"
    if ls $pred_file 1> /dev/null 2>&1; then
        echo "🎯 PRÉDICTIONS PM2.5 TROUVÉES!"
        python3 << 'PYEOF'
import numpy as np
import glob
import os

results_dir = "/scratch/project_462000640/ammar/rossice/results"
pred_files = glob.glob(f"{results_dir}/pm25_predictions_*.npz")

if pred_files:
    latest_pred = sorted(pred_files)[-1]
    print(f"📁 Fichier: {os.path.basename(latest_pred)}")
    
    data = np.load(latest_pred)
    print("\n📊 Contenu:")
    for key in data.files:
        arr = data[key]
        print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")
        if 'predictions' in key or 'pm25' in key:
            print(f"    Min: {arr.min():.2f}, Max: {arr.max():.2f}, Mean: {arr.mean():.2f}")
    
    # Statistiques de performance si disponibles
    if 'mae' in data.files:
        print(f"\n📈 Performance:")
        print(f"  - MAE: {data['mae'].item():.4f}")
    if 'rmse' in data.files:
        print(f"  - RMSE: {data['rmse'].item():.4f}")
PYEOF
    fi
}

# Boucle principale de monitoring
COUNTER=0
while true; do
    clear
    echo "🤖 ROSSICE AUTO-MONITOR - Cycle #$((++COUNTER))"
    echo "🕐 $(date)"
    echo "=" | tr = '='*60
    
    # Vérifier si le job est toujours actif
    if check_job_status; then
        JOB_INFO=$(squeue -j $JOB_ID --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R")
        echo "✅ JOB ACTIF:"
        echo "$JOB_INFO"
        echo ""
        
        # Si le job est en cours d'exécution (pas en attente)
        if ! squeue -j $JOB_ID | grep -q "PD"; then
            extract_metrics
            check_checkpoints
        else
            echo "⏳ Job en attente de ressources..."
        fi
    else
        echo "🏁 JOB TERMINÉ!"
        
        # Analyser les logs finaux
        echo -e "\n📋 RÉSUMÉ FINAL:"
        tail -50 "$LOG_DIR/climax_caqra_${JOB_ID}.out" 2>/dev/null | grep -E "completed|finished|error|fail|success|mae|rmse|test"
        
        # Vérifier les résultats
        check_checkpoints
        analyze_predictions
        
        # Créer un rapport final
        REPORT_FILE="$RESULTS_DIR/rossice_report_$(date +%Y%m%d_%H%M%S).txt"
        {
            echo "RAPPORT FINAL ROSSICE PM2.5"
            echo "=========================="
            echo "Date: $(date)"
            echo "Job ID: $JOB_ID"
            echo ""
            echo "CHECKPOINTS:"
            ls -la $CHECKPOINT_DIR/rossice_*
            echo ""
            echo "PRÉDICTIONS:"
            ls -la $RESULTS_DIR/pm25_*
            echo ""
            echo "DERNIÈRES LIGNES DU LOG:"
            tail -100 "$LOG_DIR/climax_caqra_${JOB_ID}.out"
        } > "$REPORT_FILE"
        
        echo -e "\n📄 Rapport sauvegardé: $REPORT_FILE"
        echo "✅ Monitoring terminé!"
        break
    fi
    
    # Attendre avant la prochaine vérification
    echo -e "\n⏰ Prochaine vérification dans 60 secondes..."
    sleep 60
done

echo -e "\n🎉 FIN DU MONITORING AUTOMATIQUE"
