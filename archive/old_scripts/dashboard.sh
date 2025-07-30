#!/bin/bash
# Tableau de bord ROSSICE

clear
echo "╔══════════════════════════════════════════════════════════╗"
echo "║             🚀 ROSSICE PM2.5 DASHBOARD                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "📅 $(date)"
echo ""

# Status du job
echo "📊 STATUS DU JOB:"
squeue -j 12049899 2>/dev/null || echo "   Job terminé ou non trouvé"
echo ""

# Processus en cours
echo "⚙️  PROCESSUS ACTIFS:"
ps aux | grep -E "rossice|monitor|automation" | grep -v grep | wc -l | xargs -I {} echo "   {} processus de monitoring/automation actifs"
echo ""

# Checkpoints
echo "💾 CHECKPOINTS:"
ls -la /scratch/project_462000640/ammar/rossice/checkpoints/rossice_* 2>/dev/null | wc -l | xargs -I {} echo "   {} checkpoints trouvés"
echo ""

# Logs
echo "📋 DERNIÈRES ACTIVITÉS:"
tail -5 /scratch/project_462000640/ammar/rossice/monitor_output.log 2>/dev/null | grep -v "^$" | sed 's/^/   /'
echo ""

# Espace disque
echo "💿 ESPACE DISQUE:"
df -h /scratch/project_462000640/ammar/rossice | tail -1 | awk '{print "   Utilisé: "$3" / "$2" ("$5")"}'
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🤖 Tout est automatisé ! Les résultats seront prêts     ║"
echo "║     dès que l'entraînement sera terminé.                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
