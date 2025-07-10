#!/usr/bin/env python3
"""
Script d'entraînement ClimaX pour prédiction de polluants CAQRA avec fine-tuning
Utilise le checkpoint pré-entraîné checkpoints/climax_1.40625deg.ckpt
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Ajouter le chemin vers ClimaX
sys.path.append('/scratch/project_462000640/ammar/rossice/climax/src')

# Imports ClimaX
try:
    from climax.arch import ClimaXVisionTransformer
    from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
    from climax.utils.metrics import lat_weighted_mse_loss, lat_weighted_rmse, lat_weighted_mae
except ImportError as e:
    print(f"Erreur import ClimaX: {e}")
    print("Vérifiez que le repo ClimaX est cloné dans /scratch/project_462000640/ammar/rossice/