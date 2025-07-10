#!/bin/bash
#SBATCH --job-name=climax_caqra_finetune
#SBATCH --account=project_462000640
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/project_462000640/ammar/rossice/logs/climax_caqra_%j.out
#SBATCH --error=/scratch/project_462000640/ammar/rossice/logs/climax_caqra_%j.err

# Configuration des modules pour Lumi
module load LUMI/22.08
module load partition/G
module load rocm/5.0.2
module load PyTorch/1.12.1-rocm-5.0.2-python-3.10-singularity-20220712

# Variables d'environnement pour ROCm
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Variables d'environnement pour PyTorch Lightning et DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn

# Optimisations mémoire
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=7

# Aller dans le répertoire de travail
cd /scratch/project_462000640/ammar/rossice/

# Créer dossiers nécessaires
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs
mkdir -p outputs/predictions

# Afficher info système
echo "=== Information Système ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "GPU Info:"
rocm-smi
echo "Python version: $(python --version)"
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo "CUDA/ROCm available:"
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo "GPU count:"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
echo "=========================="

# Vérifier structure des dossiers
echo "=== Vérification structure ==="
echo "Dossier rossice:"
ls -la /scratch/project_462000640/ammar/rossice/
echo "Dossier data_rossice:"
ls -la /scratch/project_462000640/ammar/data_rossice/ | head -10
echo "Checkpoint pré-entraîné:"
ls -la /scratch/project_462000640/ammar/rossice/checkpoints/
echo "=========================="

# Installer dépendances supplémentaires si nécessaire
echo "=== Installation dépendances ==="
pip install --user xarray netcdf4 pytorch-lightning tensorboard pyyaml scipy
pip install --user timm einops  # Dépendances spécifiques à ClimaX
echo "Dépendances installées"

# Vérifier que climax est disponible
echo "=== Vérification climax ==="
if [ -d "/scratch/project_462000640/ammar/rossice/climax" ]; then
    echo "✓ Dossier climax trouvé"
    ls -la /scratch/project_462000640/ammar/rossice/climax/
else
    echo "✗ Dossier climax non trouvé - clonage du repo"
    cd /scratch/project_462000640/ammar/rossice/
    git clone https://github.com/microsoft/ClimaX.git
    mv ClimaX climax  # Renommer pour cohérence
    echo "✓ climax cloné et renommé"
fi

# Test rapide du dataloader avant entraînement complet
echo "=== Test du dataloader CAQRA ==="
cd /scratch/project_462000640/ammar/rossice/
python -c "
import sys
sys.path.append('data')
sys.path.append('ClimaX/src')
from caqra_dataloader import CAQRADataset
import time

print('Test création dataset CAQRA...')
start_time = time.time()
try:
    dataset = CAQRADataset(
        data_path='/scratch/project_462000640/ammar/data_rossice/',
        years=[2013],
        time_history=3,
        time_future=6,
        spatial_subsample=4,  # Sous-échantillonnage pour test rapide
        target_resolution=(64, 128),  # Résolution réduite pour test
        normalize=True
    )
    print(f'Dataset créé en {time.time() - start_time:.2f}s')
    print(f'Taille dataset: {len(dataset)}')

    if len(dataset) > 0:
        print('Test chargement échantillon...')
        start_time = time.time()
        input_tensor, target_tensor = dataset[0]
        print(f'Échantillon chargé en {time.time() - start_time:.2f}s')
        print(f'Input shape: {input_tensor.shape}')
        print(f'Target shape: {target_tensor.shape}')
        print('✓ Test dataloader réussi !')
    else:
        print('✗ ERREUR: Dataset vide')
        exit(1)
        
except Exception as e:
    print(f'✗ ERREUR: Test dataloader échoué: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "✗ ERREUR: Test dataloader échoué"
    exit 1
fi

echo "=== Test du modèle climax ==="
python -c "
import sys
sys.path.append('climax/src')
import torch

try:
    from climax.arch import ClimaX
    
    print('Test création modèle climax...')
    model = ClimaX(
        default_vars=['u', 'v', 'temp', 'rh', 'psfc'],
        img_size=[64, 128],
        patch_size=4,
        embed_dim=512,
        depth=12,
        num_heads=8
    )
    
    # Test forward
    x = torch.randn(1, 3, 5, 64, 128)  # (batch, time, channels, H, W)
    output = model(x)
    print(f'✓ Test modèle réussi: {x.shape} -> {output.shape}')
    
except Exception as e:
    print(f'✗ ERREUR: Test modèle échoué: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "✗ ERREUR: Test modèle échoué"
    exit 1
fi

# Validation de la configuration
echo "=== Validation configuration ==="
if [ -f "configs/caqra_pollution_finetune.yaml" ]; then
    echo "✓ Configuration trouvée"
    python -c "
import yaml
try:
    with open('configs/caqra_pollution_finetune.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ Configuration YAML valide')
    print(f'Checkpoint: {config[\"pretrained\"][\"checkpoint_path\"]}')
    
    # Vérifier que le checkpoint existe
    import os
    checkpoint_path = config['pretrained']['checkpoint_path']
    if os.path.exists(checkpoint_path):
        print(f'✓ Checkpoint pré-entraîné trouvé: {checkpoint_path}')
        
        # Afficher taille du checkpoint
        size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
        print(f'Taille checkpoint: {size_mb:.1f} MB')
    else:
        print(f'✗ Checkpoint non trouvé: {checkpoint_path}')
        exit(1)
        
except Exception as e:
    print(f'✗ Erreur configuration: {e}')
    exit(1)
"
else
    echo "✗ Configuration non trouvée: configs/caqra_pollution_finetune.yaml"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "✗ ERREUR: Validation configuration échouée"
    exit 1
fi

# Validation optionnelle des données avant entraînement
echo "=== Validation complète des données ==="
python scripts/train_caqra_finetune.py \
    --config configs/caqra_pollution_finetune.yaml \
    --validate_data

if [ $? -ne 0 ]; then
    echo "✗ ERREUR: Validation des données échouée"
    exit 1
fi

echo "=== Début entraînement ClimaX CAQRA Fine-tuning ==="

# Lancer l'entraînement avec fine-tuning
python scripts/train_caqra_finetune.py \
    --config configs/caqra_pollution_finetune.yaml

echo "=== Fin du job ==="
echo "Date fin: $(date)"

# Afficher statistiques du job
echo "=== Statistiques du job ==="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State

# Afficher résultats finaux
echo "=== Résultats ==="
echo "Checkpoints sauvegardés:"
ls -la /scratch/project_462000640/ammar/rossice/checkpoints/climax_caqra_finetune* 2>/dev/null || echo "Aucun checkpoint trouvé"

echo "Logs tensorboard:"
ls -la /scratch/project_462000640/ammar/rossice/logs/ 2>/dev/null || echo "Aucun log trouvé"

echo "Prédictions:"
ls -la /scratch/project_462000640/ammar/rossice/outputs/predictions/ 2>/dev/null || echo "Aucune prédiction trouvée"