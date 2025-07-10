#!/bin/bash
# Script de lancement des tests ClimaX CAQRA

echo "🚀 LANCEMENT DES TESTS CLIMAX CAQRA"
echo "=================================="

# Aller dans le bon répertoire
cd /scratch/project_462000640/ammar/rossice/

# Vérifier que nous sommes au bon endroit
echo "📁 Répertoire de travail: $(pwd)"

# Vérifier les fichiers nécessaires
echo "🔍 Vérification des fichiers..."

# climax (attention à la casse)
if [ -d "climax" ]; then
    echo "  climax trouvé"
else
    echo "  climax manquant - clonage..."
    git clone https://github.com/microsoft/ClimaX.git
    mv ClimaX climax  # Renommer pour cohérence
fi

# Dataloader
if [ -f "data/caqra_dataloader.py" ]; then
    echo "   Dataloader CAQRA trouvé"
else
    echo "   Dataloader CAQRA manquant"
    exit 1
fi

# Configuration
if [ -f "configs/caqra_pollution_finetune.yaml" ]; then
    echo "   Configuration trouvée"
else
    echo "   Configuration manquante"
    exit 1
fi

# Checkpoint
if [ -f "checkpoints/climax_1.40625deg.ckpt" ]; then
    echo "   Checkpoint pré-entraîné trouvé"
else
    echo "  Checkpoint pré-entraîné manquant"
    exit 1
fi

# Données
if [ -d "/scratch/project_462000640/ammar/data_rossice" ]; then
    echo "  Données CAQRA trouvées"
else
    echo "   Données CAQRA manquantes"
    exit 1
fi

echo ""
echo " LANCEMENT DE LA SUITE DE TESTS"
echo "================================="

# Lancer les tests
python complete_test_suite.py

# Capturer le code de sortie
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo " TOUS LES TESTS RÉUSSIS !"
    echo ""
    echo "💡 PROCHAINES ÉTAPES :"
    echo "   1. Lancer l'entraînement: sbatch scripts/submit_lumi_finetune.sh"
    echo "   2. Monitorer les logs: tail -f logs/climax_caqra_*.out"
    echo "   3. Vérifier TensorBoard: tensorboard --logdir logs/"
else
    echo " CERTAINS TESTS ONT ÉCHOUÉ"
    echo ""
    echo " ACTIONS RECOMMANDÉES :"
    echo "   1. Vérifier les erreurs ci-dessus"
    echo "   2. Corriger les problèmes identifiés"
    echo "   3. Relancer les tests"
fi

exit $exit_code