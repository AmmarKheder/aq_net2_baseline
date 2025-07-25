#!/bin/bash -l
#SBATCH --job-name=climax_caqra_finetune
#SBATCH --account=project_462000640
#SBATCH --partition=small-g          # Partition small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # Adapté pour 4 GPUs
#SBATCH --gres=gpu:mi250:4           # 4 GPUs maximum sur small-g
#SBATCH --cpus-per-task=7
#SBATCH --mem=240G                   # Réduit proportionnellement
#SBATCH --time=72:00:00              # Durée max de 72h sur small-g
#SBATCH --output=/scratch/project_462000640/ammar/rossice/logs/climax_caqra_%j.out
#SBATCH --error=/scratch/project_462000640/ammar/rossice/logs/climax_caqra_%j.err

# Configuration des modules pour Lumi GPU
source ~/.bashrc
module purge
module load LUMI/23.09 partition/G
module load rocm/6.0.3
module load cray-python/3.11.7
echo "Modules chargés :"; module list
echo "Which Python : $(which python)"
echo "Python version : $(python --version)"

# Variables d'environnement pour ROCm
export ROCM_PATH=/opt/rocm
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Variables pour PyTorch Lightning et DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4                   # Adapté pour 4 GPUs
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn

# Optimisations mémoire
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=128
export OMP_NUM_THREADS=7

# Aller dans le répertoire de travail
cd /scratch/project_462000640/ammar/rossice/

# Créer dossiers nécessaires
mkdir -p logs checkpoints outputs outputs/predictions

# Afficher info système
echo "=== Information Système ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Modules chargés:"
module list
echo ""
echo "GPU Info:"
rocm-smi
echo ""
echo "Python test GPU:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo "=========================="

# Vérifier GPU disponible
python -c "import torch; assert torch.cuda.is_available(), 'ERREUR: GPU non détecté!'"
if [ $? -ne 0 ]; then
   echo "❌ ERREUR CRITIQUE: GPU non disponible!"
   exit 1
fi

# Installer dépendances si nécessaire
echo "=== Installation dépendances ==="
pip install --user xarray netcdf4 pytorch-lightning tensorboard pyyaml scipy timm einops
echo "Dépendances installées"

# Vérifier structure
echo "=== Vérification structure ==="
if [ ! -d "climax" ]; then
   echo "Clonage de ClimaX..."
   git clone https://github.com/microsoft/ClimaX.git
   mv ClimaX climax
fi

# 🔧 SEUL CHANGEMENT : Vérifier TON script au lieu de la config YAML
if [ ! -f "rossice_final_solution.py" ]; then
   echo "❌ ERREUR: Script rossice_final_solution.py manquant!"
   echo "Créer le fichier avec ton code Python Rossice"
   exit 1
fi

echo "✅ Structure vérifiée"

# Test rapide du dataloader
echo "=== Test rapide dataloader ==="
python -c "
import sys
sys.path.append('data')
sys.path.append('climax/src')
try:
   from caqra_dataloader import CAQRADataset
   dataset = CAQRADataset(
       data_path='/scratch/project_462000640/ammar/data_rossice/',
       years=[2013],
       time_history=3,
       time_future=6,
       spatial_subsample=8,
       target_resolution=(64, 128),
       normalize=False
   )
   print(f'✅ Dataset OK: {len(dataset)} échantillons')
except Exception as e:
   print(f'❌ Erreur dataset: {e}')
   exit(1)
"

# Test rapide du modèle
echo "=== Test rapide modèle ==="
python -c "
import sys
sys.path.append('climax/src')
import torch
try:
   from climax.arch import ClimaX
   model = ClimaX(
       default_vars=['u', 'v', 'temp', 'rh', 'psfc'],
       img_size=[64, 128],
       patch_size=4,
       embed_dim=512,
       depth=12,
       num_heads=8
   )
   print('✅ Modèle ClimaX créé')
   
   # Test GPU
   if torch.cuda.is_available():
       model = model.cuda()
       x = torch.randn(1, 5, 64, 128).cuda()
       print('✅ Modèle sur GPU')
except Exception as e:
   print(f'❌ Erreur modèle: {e}')
   exit(1)
"

echo "=== Début entraînement ClimaX CAQRA ==="
echo "Configuration: 4 GPUs, 72h max"

# 🎯 CHANGEMENT : Lancer TON script Rossice au lieu du script ClimaX original
echo "🚀 Lancement entraînement Rossice..."
python rossice_final_solution.py

echo "=== Fin du job ==="
echo "Date fin: $(date)"

# Statistiques finales
echo "=== Résultats ==="
ls -la checkpoints/rossice_final_* 2>/dev/null || echo "Pas de checkpoints Rossice"
ls -la logs/ | tail -5

echo "Job terminé - Vérifier les logs pour les détails"