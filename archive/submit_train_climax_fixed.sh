#!/bin/bash
#SBATCH --job-name=train_climax_fixed
#SBATCH --account=project_462000640
#SBATCH --partition=small-g         # ← partition small-g pour jobs longs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=4           # ← 4 GPUs au lieu de 8
#SBATCH --mem=64G
#SBATCH --time=72:00:00             # ← 72 heures au lieu de 3h
#SBATCH --output=logs/train_climax_fixed_%j.out
#SBATCH --error=logs/train_climax_fixed_%j.err

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

source venv_pytorch_rocm/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/project_462000640/ammar/rossice:$PYTHONPATH
export TIMM_FUSED_ATTN=0

# ═══ PROTECTION NETCDF ANTI-CORRUPTION ═══
echo "🛡️  Configuration protection NetCDF..."
export HDF5_USE_FILE_LOCKING=FALSE
export NETCDF4_THREAD_SAFE=1
export HDF5_DISABLE_VERSION_CHECK=2
export HDF5_USE_FILE_LOCKING=FALSE

# Protection fichiers en lecture seule
echo "🔒 Protection des fichiers NetCDF..."
find /scratch/project_462000640/ammar/data_rossice/ -name "*.nc" -exec chmod 444 {} \;
echo "✅ Fichiers protégés en lecture seule"

# Vérifier qu'aucun autre job Python ne tourne
echo "🔍 Vérification des processus Python en cours..."
ps aux | grep $USER | grep python | grep -v grep || echo "✅ Aucun processus Python concurrent"

echo "🚀 Lancement train_climax_pm25_fixed.py..."
srun python train_climax_pm25_fixed.py