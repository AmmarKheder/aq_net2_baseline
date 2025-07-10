#!/bin/bash -l
#SBATCH --job-name=rossice_back_to_working
#SBATCH --account=project_462000640
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi250:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462000640/ammar/rossice/logs/back_to_working_%j.out
#SBATCH --error=/scratch/project_462000640/ammar/rossice/logs/back_to_working_%j.err

echo "🔄 === RETOUR À LA CONFIG QUI MARCHAIT ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /scratch/project_462000640/ammar/rossice/

echo ""
echo "=== 📦 MODULES QUI MARCHAIENT AVANT ==="
module purge
module load LUMI/23.09 partition/G rocm/6.0.3 cray-python/3.11.7

echo "Modules chargés:"
module list

echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Variables d'environnement ROCm
export ROCM_PATH=/opt/rocm
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export OMP_NUM_THREADS=7

# Variables d'environnement pour Python
export PYTHONUSERBASE=/scratch/project_462000640/ammar/python_local
mkdir -p $PYTHONUSERBASE

echo ""
echo "=== 🔧 INSTALLATION PYTORCH ROCM CORRECT ==="
echo "Désinstallation de la version CUDA incorrecte..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Installation PyTorch ROCm (version compatible Lumi)..."
pip install --user --no-cache-dir \
    torch==2.4.1+rocm6.0 \
    torchvision==0.19.1+rocm6.0 \
    torchaudio==2.4.1+rocm6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.0

echo ""
echo "Installation autres dépendances..."
pip install --user --no-cache-dir \
    pytorch-lightning \
    tensorboard \
    xarray \
    netcdf4 \
    pyyaml \
    scipy \
    timm \
    einops

echo ""
echo "=== ✅ VÉRIFICATION PYTORCH ROCM ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'GPU disponible: {torch.cuda.is_available()}')
print(f'Nombre de GPUs: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        
    # Test simple allocation
    print('Test allocation GPU...')
    x = torch.randn(100, 100).cuda()
    y = x + 1
    print(f'✅ Test GPU allocation réussi: {x.shape}')
    print(f'GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB')
else:
    print('❌ ERREUR: GPU non détecté')
    
    # Debug info
    print('Debug info:')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'ROCm available: {hasattr(torch.cuda, \"is_available\") and torch.cuda.is_available()}')
"

if [ $? -ne 0 ]; then
    echo "❌ Erreur PyTorch"
    exit 1
fi

echo ""
echo "=== 🧪 TEST GPU SIMPLE ==="
python -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'Device actuel: {device}')
    print(f'Device name: {torch.cuda.get_device_name(device)}')
    
    # Test calcul
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x.T)
    print(f'✅ Test calcul GPU: {x.shape} @ {x.T.shape} = {y.shape}')
    print(f'Result sample: {y[0,0].item():.4f}')
else:
    print('❌ Pas de GPU pour test calcul')
    exit(1)
"

echo ""
echo "=== 📊 TEST DATASET RAPIDE ==="
python -c "
import sys
sys.path.append('/scratch/project_462000640/ammar/rossice/data')

try:
    from caqra_dataloader import CAQRADataset
    
    print('Création dataset test...')
    dataset = CAQRADataset(
        data_path='/scratch/project_462000640/ammar/data_rossice/',
        years=[2013],
        time_history=3,
        time_future=6,
        spatial_subsample=16,  # Plus grand pour test rapide
        target_resolution=(32, 64),
        normalize=False
    )
    
    print(f'✅ Dataset: {len(dataset)} échantillons')
    
    if len(dataset) > 0:
        inputs, targets = dataset[0]
        print(f'✅ Sample: inputs={inputs.shape}, targets={targets.shape}')
        print(f'Input range: [{inputs.min():.3f}, {inputs.max():.3f}]')
        print(f'Target range: [{targets.min():.3f}, {targets.max():.3f}]')
    else:
        print('❌ Dataset vide')
        
except Exception as e:
    print(f'⚠️  Dataset error: {e}')
    # Pas d'exit ici, continuer quand même
"

echo ""
echo "=== 🧠 TEST MODÈLE CLIMAX SIMPLE ==="
python -c "
import sys
sys.path.append('/scratch/project_462000640/ammar/rossice/climax/src')
import torch

try:
    from climax.arch import ClimaX
    
    print('Création modèle ClimaX...')
    model = ClimaX(
        default_vars=['u', 'v', 'temp', 'rh', 'psfc'],
        img_size=[32, 64],
        patch_size=4,
        embed_dim=128,  # Réduit pour test
        depth=2,        # Réduit pour test
        num_heads=4
    )
    
    print('✅ Modèle créé')
    
    # Test GPU
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(1, 5, 32, 64).cuda()
        print(f'✅ Modèle et input sur GPU: {x.shape}')
        
        model.eval()
        with torch.no_grad():
            try:
                print('Test forward simple...')
                output = model(x)
                print(f'✅ Forward simple: {x.shape} -> {output.shape}')
            except Exception as forward_error:
                print(f'Forward simple échoué: {forward_error}')
                print('Test forward avec arguments...')
                
                batch_size = x.shape[0]
                y = torch.zeros(batch_size, 6, 32, 64).cuda()
                lead_times = torch.zeros(batch_size).cuda()
                variables = ['u', 'v', 'temp', 'rh', 'psfc']
                out_variables = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
                lat = torch.zeros(32, 64).cuda()
                
                try:
                    result = model(x, y, lead_times, variables, out_variables, None, lat)
                    output = result[1] if isinstance(result, tuple) else result
                    print(f'✅ Forward avec args: {x.shape} -> {output.shape}')
                except Exception as e:
                    print(f'❌ Forward avec args échoué: {e}')
                    raise
    else:
        print('❌ Pas de GPU pour test modèle')
        
except Exception as e:
    print(f'❌ Erreur modèle: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=== 🎯 RÉSUMÉ FINAL ==="
echo "Date: $(date)"
echo ""

# Test final pour confirmer
python -c "
import torch
gpu_ok = torch.cuda.is_available()
print(f'GPU Status: {\"✅ OK\" if gpu_ok else \"❌ FAIL\"}')

if gpu_ok:
    print('🎉 === CONFIGURATION FONCTIONNELLE ===')
    print('✅ Modules Lumi: OK')
    print('✅ PyTorch ROCm: OK') 
    print('✅ GPU détection: OK')
    print('🚀 PRÊT POUR ENTRAÎNEMENT ROSSICE!')
else:
    print('❌ === PROBLÈME PERSISTANT ===')
    print('GPU toujours non détecté')
"

echo ""
echo "🏁 FIN DU TEST - Configuration Original Restaurée"