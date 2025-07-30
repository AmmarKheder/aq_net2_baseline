#!/bin/bash

JOB_ID=12050352
LOG_FILE="logs/rossice_corrected_${JOB_ID}.out"
ERR_FILE="logs/rossice_corrected_${JOB_ID}.err"

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                 🚀 ROSSICE PM2.5 - MONITORING LIVE               ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📅 $(date)"
    echo "🆔 Job ID: $JOB_ID"
    echo ""
    
    # État du job
    STATUS=$(squeue -j $JOB_ID -h -o "%T %M %l" 2>/dev/null)
    if [ -n "$STATUS" ]; then
        STATE=$(echo $STATUS | awk '{print $1}')
        TIME=$(echo $STATUS | awk '{print $2}')
        LIMIT=$(echo $STATUS | awk '{print $3}')
        echo "✅ Job actif: $STATE (Temps: $TIME / Limite: $LIMIT)"
    else
        echo "🏁 Job terminé"
        if [ -f "results/TRAINING_SUCCESS.txt" ]; then
            echo "✅ SUCCÈS!"
            cat results/TRAINING_SUCCESS.txt
        fi
    fi
    
    # Progression
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "📊 PROGRESSION:"
        
        # Phase actuelle
        if grep -q "Starting training" "$LOG_FILE"; then
            echo "🏃 Phase: ENTRAÎNEMENT"
            
            # Dernière epoch
            EPOCH=$(grep -E "Epoch [0-9]+/" "$LOG_FILE" | tail -1)
            if [ -n "$EPOCH" ]; then
                echo "   $EPOCH"
                
                # Métriques
                METRICS=$(grep -A3 "$EPOCH" "$LOG_FILE" | grep -E "Loss:|MAE:|RMSE:" | tail -3)
                if [ -n "$METRICS" ]; then
                    echo "$METRICS" | sed 's/^/   /'
                fi
            fi
            
            # Progression des batchs
            BATCH=$(grep -E "Batch [0-9]+/" "$LOG_FILE" | tail -1)
            [ -n "$BATCH" ] && echo "   $BATCH"
            
        elif grep -q "Chargement des données" "$LOG_FILE"; then
            echo "📁 Phase: CHARGEMENT DES DONNÉES"
        elif grep -q "Installing packages" "$LOG_FILE"; then
            echo "📦 Phase: INSTALLATION"
            PKG=$(tail -5 "$LOG_FILE" | grep -E "Collecting|Installing|Successfully" | tail -1)
            [ -n "$PKG" ] && echo "   $PKG"
        fi
    fi
    
    # Erreurs
    if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
        echo ""
        echo "⚠️  ERREURS DÉTECTÉES:"
        tail -5 "$ERR_FILE" | sed 's/^/   /'
    fi
    
    # Résultats
    echo ""
    echo "📁 RÉSULTATS:"
    if ls results/pm25_predictions*.npz 2>/dev/null; then
        echo "   ✅ Prédictions disponibles!"
        ls -lh results/pm25_predictions*.npz | tail -1 | sed 's/^/   /'
    else
        echo "   ⏳ En attente..."
    fi
    
    echo ""
    echo "Actualisation dans 10 secondes... (Ctrl+C pour quitter)"
    sleep 10
done
