#!/usr/bin/env python3
"""
Rossice - Script final fonctionnel
Correction du problème de forward ClimaX
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Paths
sys.path.insert(0, "/scratch/project_462000640/ammar/rossice/climax/src")
sys.path.insert(0, "/scratch/project_462000640/ammar/rossice/data")

print("🎯 === ROSSICE FINAL WORKING ===")
print(f"Date: {datetime.now()}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)")

# Configuration
CONFIG = {
    'data_path': '/scratch/project_462000640/ammar/data_rossice/',
    'train_years': [2013],
    'val_years': [2014],
    'input_vars': ['u', 'v', 'temp', 'rh', 'psfc'],
    'output_vars': ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3'],
    'time_history': 3,
    'time_future': 6,
    'resolution': [32, 64],
    'batch_size': 2,
    'max_epochs': 3,
    'lr': 1e-3
}

class RossiceClimaXWrapper(nn.Module):
    """Wrapper pour ClimaX qui gère les arguments automatiquement"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Import et création ClimaX
        from climax.arch import ClimaX
        
        self.climax = ClimaX(
            default_vars=config['input_vars'],
            img_size=config['resolution'],
            patch_size=4,
            embed_dim=256,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0
        )
        
        # Adapter la tête pour les polluants
        self._adapt_head()
        print("✅ ClimaX adapté pour pollution")
    
    def _adapt_head(self):
        """Adapte la tête pour prédire les polluants"""
        num_pollutants = len(self.config['output_vars'])
        patch_size = 4
        
        # Calculer nouvelle dimension de sortie
        output_dim = num_pollutants * (patch_size ** 2)
        
        if hasattr(self.climax, 'head'):
            if isinstance(self.climax.head, nn.Sequential):
                in_features = self.climax.head[-1].in_features
                self.climax.head[-1] = nn.Linear(in_features, output_dim)
            else:
                in_features = self.climax.head.in_features
                self.climax.head = nn.Linear(in_features, output_dim)
            
            print(f"Tête adaptée: {in_features} → {output_dim} (pour {num_pollutants} polluants)")
    
    def forward(self, x):
        """Forward qui gère automatiquement les arguments ClimaX"""
        batch_size = x.shape[0]
        device = x.device
        H, W = x.shape[-2:]
        
        # Créer les arguments requis par ClimaX
        y = torch.zeros(batch_size, len(self.config['output_vars']), H, W).to(device)
        lead_times = torch.zeros(batch_size).to(device)
        variables = self.config['input_vars']
        out_variables = self.config['output_vars']
        metric = None  # Pas de métrique pour forward
        lat = torch.zeros(H, W).to(device)
        
        # Appel ClimaX avec tous les arguments
        result = self.climax(x, y, lead_times, variables, out_variables, metric, lat)
        
        # Extraire les prédictions (ClimaX retourne un tuple)
        if isinstance(result, tuple) and len(result) >= 2:
            predictions = result[1]  # Les prédictions sont en position 1
        else:
            predictions = result
        
        return predictions

class RossiceLightningModule(pl.LightningModule):
    """Module Lightning pour Rossice"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RossiceClimaXWrapper(config)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs)
        
        # Adapter les shapes si nécessaire
        if predictions.shape != targets.shape:
            # Reshape predictions pour matcher targets
            if predictions.dim() == 2:  # (batch, features)
                batch_size = predictions.shape[0]
                predictions = predictions.view(batch_size, len(self.config['output_vars']), 
                                             self.config['resolution'][0], self.config['resolution'][1])
        
        loss = nn.functional.mse_loss(predictions, targets)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Métriques additionnelles
        with torch.no_grad():
            rmse = torch.sqrt(loss)
            self.log('train_rmse', rmse, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs)
        
        # Adapter shapes
        if predictions.shape != targets.shape:
            if predictions.dim() == 2:
                batch_size = predictions.shape[0]
                predictions = predictions.view(batch_size, len(self.config['output_vars']), 
                                             self.config['resolution'][0], self.config['resolution'][1])
        
        loss = nn.functional.mse_loss(predictions, targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        with torch.no_grad():
            rmse = torch.sqrt(loss)
            self.log('val_rmse', rmse, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config['lr'],
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['max_epochs']
        )
        
        return [optimizer], [scheduler]

def main():
    """Fonction principale"""
    print("📊 === CRÉATION DES DATASETS ===")
    
    # Import dataset
    from caqra_dataloader import CAQRADataset
    
    # Dataset train
    train_dataset = CAQRADataset(
        data_path=CONFIG['data_path'],
        years=CONFIG['train_years'],
        time_history=CONFIG['time_history'],
        time_future=CONFIG['time_future'],
        spatial_subsample=8,
        target_resolution=tuple(CONFIG['resolution']),
        normalize=False
    )
    
    # Dataset val (plus petit pour test)
    val_dataset = CAQRADataset(
        data_path=CONFIG['data_path'],
        years=CONFIG['val_years'],
        time_history=CONFIG['time_history'],
        time_future=CONFIG['time_future'],
        spatial_subsample=16,  # Plus rapide
        target_resolution=tuple(CONFIG['resolution']),
        normalize=False
    )
    
    print(f"✅ Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ DataLoaders: Train={len(train_loader)} batches, Val={len(val_loader)} batches")
    
    print("\n🧠 === CRÉATION DU MODÈLE ===")
    model = RossiceLightningModule(CONFIG)
    
    print("\n⚡ === TEST FORWARD ===")
    # Test avec un batch
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
    
    print(f"Sample batch: inputs={inputs.shape}, targets={targets.shape}")
    
    # Mettre sur GPU si disponible
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        print("✅ Modèle et données sur GPU")
    
    # Test forward
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        print(f"✅ Forward réussi: {inputs.shape} → {predictions.shape}")
        print(f"Target shape: {targets.shape}")
        
        # Test loss
        if predictions.shape == targets.shape:
            loss = nn.functional.mse_loss(predictions, targets)
            print(f"✅ Loss calculée: {loss.item():.6f}")
        else:
            print(f"⚠️  Shape mismatch: pred={predictions.shape} vs target={targets.shape}")
    
    print("\n🏃 === ENTRAÎNEMENT ===")
    
    # Callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='/scratch/project_462000640/ammar/rossice/checkpoints/',
        filename='rossice_final_{epoch:02d}_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Logger
    from pytorch_lightning.loggers import TensorBoardLogger
    
    logger = TensorBoardLogger(
        save_dir='/scratch/project_462000640/ammar/rossice/logs/',
        name='rossice_final'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        gradient_clip_val=1.0
    )
    
    print(f"✅ Trainer configuré pour {CONFIG['max_epochs']} epochs")
    
    # Entraînement
    print("🚀 Début de l'entraînement...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n🎉 === ENTRAÎNEMENT TERMINÉ ===")
    print(f"Checkpoints sauvegardés dans: /scratch/project_462000640/ammar/rossice/checkpoints/")
    print(f"Logs TensorBoard dans: /scratch/project_462000640/ammar/rossice/logs/rossice_final/")
    
    # Test final
    print("\n📊 === TEST FINAL ===")
    trainer.test(model, val_loader)
    
    print("✅ ROSSICE ENTRAÎNEMENT COMPLET RÉUSSI!")

if __name__ == "__main__":
    main()

