#!/bin/bash
clear
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    🚀 ROSSICE PM2.5 - ÉTAT EN DIRECT             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📅 $(date)"
echo "🆔 Job ID: 12050327"
echo ""

# État du job
STATUS=$(squeue -j 12050327 -h -o "%T %M" 2>/dev/null)
if [ -n "$STATUS" ]; then
    echo "✅ Job actif : $STATUS"
else
    echo "🏁 Job terminé"
fi

# Phase actuelle
LOG="/scratch/project_462000640/ammar/rossice/logs/rossice_pm25_final_12050327.out"
if [ -f "$LOG" ]; then
    if grep -q "Starting training" "$LOG"; then
        echo "🏃 Phase: ENTRAÎNEMENT EN COURS"
        # Chercher l'epoch actuelle
        EPOCH=$(grep -E "Epoch [0-9]+" "$LOG" | tail -1)
        [ -n "$EPOCH" ] && echo "   $EPOCH"
    elif grep -q "Installing packages" "$LOG"; then
        echo "📦 Phase: INSTALLATION DES DÉPENDANCES"
        # Montrer le dernier paquet
        PKG=$(tail -20 "$LOG" | grep -E "Collecting|Installing|Successfully" | tail -1)
        [ -n "$PKG" ] && echo "   $PKG"
    elif grep -q "Testing installation" "$LOG"; then
        echo "🧪 Phase: TEST GPU"
    elif grep -q "Generating reports" "$LOG"; then
        echo "📊 Phase: GÉNÉRATION DES RÉSULTATS"
    fi
fi

# Métriques si disponibles
if [ -f "$LOG" ] && grep -q "loss" "$LOG"; then
    echo ""
    echo "📈 DERNIÈRES MÉTRIQUES:"
    grep -E "loss|rmse|mae" "$LOG" | tail -3 | sed 's/^/   /'
fi

# Résultats
echo ""
echo "📁 RÉSULTATS:"
if ls /scratch/project_462000640/ammar/rossice/results/pm25_predictions*.npz 2>/dev/null; then
    echo "   ✅ Prédictions PM2.5 disponibles!"
    ls -lh /scratch/project_462000640/ammar/rossice/results/pm25_predictions*.npz | tail -1
else
    echo "   ⏳ En attente..."
fi

if [ -f "/scratch/project_462000640/ammar/rossice/results/TRAINING_SUCCESS.txt" ]; then
    echo ""
    echo "🎉 SUCCÈS! L'entraînement est terminé!"
fi
