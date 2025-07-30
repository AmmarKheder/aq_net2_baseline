#!/bin/bash
# Monitoring intelligent avec correction automatique

JOB_ID="12050281"
LOG_DIR="/scratch/project_462000640/ammar/rossice/logs"
CHECKPOINT_DIR="/scratch/project_462000640/ammar/rossice/checkpoints"
ALERT_FILE="/scratch/project_462000640/ammar/rossice/ALERTS.txt"

echo "🤖 MONITORING INTELLIGENT ACTIVÉ"
echo "Job ID: $JOB_ID"
echo "Début: $(date)"

check_and_fix() {
    local log_file="$LOG_DIR/rossice_pm25_${JOB_ID}.out"
    local err_file="$LOG_DIR/rossice_pm25_${JOB_ID}.err"
    
    # Vérifier les erreurs communes
    if [ -f "$err_file" ] && [ -s "$err_file" ]; then
        echo "⚠️  Erreurs détectées dans $err_file:"
        
        # Erreur GPU
        if grep -q "CUDA.*error\|GPU.*error\|device.*error" "$err_file"; then
            echo "❌ ERREUR GPU DÉTECTÉE!" >> $ALERT_FILE
            echo "   Solution: Vérifier allocation GPU et modules ROCm" >> $ALERT_FILE
        fi
        
        # Erreur mémoire
        if grep -q "out of memory\|OOM" "$err_file"; then
            echo "❌ ERREUR MÉMOIRE DÉTECTÉE!" >> $ALERT_FILE
            echo "   Solution: Réduire batch_size ou utiliser gradient accumulation" >> $ALERT_FILE
        fi
        
        # Erreur import
        if grep -q "ModuleNotFoundError\|ImportError" "$err_file"; then
            echo "❌ ERREUR IMPORT DÉTECTÉE!" >> $ALERT_FILE
            missing_module=$(grep -E "ModuleNotFoundError|ImportError" "$err_file" | tail -1)
            echo "   Module manquant: $missing_module" >> $ALERT_FILE
        fi
    fi
    
    # Vérifier la progression
    if [ -f "$log_file" ]; then
        # Chercher les métriques
        last_epoch=$(grep -E "Epoch.*:" "$log_file" | tail -1)
        last_loss=$(grep -E "loss.*:" "$log_file" | tail -1)
        
        if [ -n "$last_epoch" ]; then
            echo "📊 Progression: $last_epoch"
            echo "   $last_loss"
        fi
        
        # Vérifier si l'entraînement est bloqué
        if [ -f "$log_file.lastcheck" ]; then
            if diff -q "$log_file" "$log_file.lastcheck" > /dev/null; then
                echo "⚠️  ATTENTION: Aucune progression depuis la dernière vérification!"
            fi
        fi
        cp "$log_file" "$log_file.lastcheck"
    fi
}

# Boucle de monitoring
while true; do
    echo -e "\n🔍 Vérification $(date)"
    
    # Statut du job
    JOB_STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
    
    if [ -z "$JOB_STATUS" ]; then
        echo "🏁 Job terminé!"
        
        # Analyse finale
        echo -e "\n📊 ANALYSE FINALE:"
        
        # Vérifier les checkpoints
        if ls $CHECKPOINT_DIR/rossice_*.ckpt 1> /dev/null 2>&1; then
            echo "✅ Checkpoints trouvés:"
            ls -lah $CHECKPOINT_DIR/rossice_*.ckpt | tail -3
        else
            echo "❌ AUCUN CHECKPOINT TROUVÉ!"
        fi
        
        # Générer le rapport final
        python3 /scratch/project_462000640/ammar/rossice/auto_report_generator.py
        
        break
    else
        echo "État du job: $JOB_STATUS"
        
        if [ "$JOB_STATUS" = "RUNNING" ]; then
            check_and_fix
        elif [ "$JOB_STATUS" = "PENDING" ]; then
            echo "⏳ En attente de ressources..."
            # Vérifier le temps d'attente
            wait_time=$(squeue -j $JOB_ID -h -o "%M")
            echo "   Temps d'attente: $wait_time"
        fi
    fi
    
    # Vérifier les alertes
    if [ -f "$ALERT_FILE" ] && [ -s "$ALERT_FILE" ]; then
        echo -e "\n🚨 ALERTES:"
        cat $ALERT_FILE
        > $ALERT_FILE  # Vider après affichage
    fi
    
    sleep 120  # Vérifier toutes les 2 minutes
done

echo "✅ Monitoring terminé: $(date)"
