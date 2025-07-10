#!/usr/bin/env python3
"""Script d'entraînement principal pour Rossice"""

import os
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Setup paths
sys.path.insert(0, "/scratch/project_462000640/ammar/rossice/climax/src")
sys.path.insert(0, "/scratch/project_462000640/ammar/rossice/data")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config YAML')
    parser.add_argument('--test', action='store_true', help='Mode test')
    args = parser.parse_args()
    
    # Charger config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"🚀 Rossice Training - Config: {args.config}")
    
    # TODO: Implémenter le modèle et l'entraînement
    # (Utiliser le code des artifacts précédents)
    
    print("✅ Terminé")

if __name__ == "__main__":
    main()
