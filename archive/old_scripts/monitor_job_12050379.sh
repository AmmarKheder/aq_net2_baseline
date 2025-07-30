#!/bin/bash

JOB_ID=12050379
LOG_FILE="logs/rossice_opt_${JOB_ID}.out"
ERR_FILE="logs/rossice_opt_${JOB_ID}.err"

echo "📊 Surveillance du job $JOB_ID - Appuyez sur Ctrl+C pour arrêter"
echo ""

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║               🚀 ROSSICE PM2.5 - JOB $JOB_ID                    ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "⏰ $(date)"
    
    # État du job
    STATUS=$(squeue -j $JOB_ID -h -o "%T %M %l" 2>/dev/null)
    if [ -n "$STATUS" ]; then
        echo "📍 État: $(echo $STATUS | awk '{print $1}') | Temps: $(echo $STATUS | awk '{print $2}')"
    else
        echo "✅ Job terminé"
        # Vérifier si succès
        if [ -f "results/TRAINING_SUCCESS.txt" ]; then
            echo ""
            echo "🎉 ENTRAÎNEMENT RÉUSSI!"
            cat results/TRAINING_SUCCESS.txt
        fi
    fi
    
    # Dernières lignes du log
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "📜 Dernières activités:"
        echo "────────────────────────────────────────────────────────────"
        tail -15 "$LOG_FILE" | sed 's/^/  /'
    fi
    
    # Erreurs
    if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
        echo ""
        echo "⚠️  Dernières erreurs:"
        echo "────────────────────────────────────────────────────────────"
        tail -5 "$ERR_FILE" | sed 's/^/  /'
    fi
    
    sleep 5
done
