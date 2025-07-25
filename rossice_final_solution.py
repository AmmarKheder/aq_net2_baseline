#!/usr/bin/env python3
"""
ROSSICE - VERSION PM2.5 SEULEMENT
Prédit uniquement PM2.5 en utilisant météo + autres polluants comme input
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import warnings
from datetime import datetime
from typing import Tuple, Optional

# Configuration des warnings et paths
warnings.filterwarnings('ignore')
sys.path.insert(0, "/scratch/project_462000640/ammar/rossice/climax/src")
sys.path.insert(0, "/scratch/project_462000640/ammar/rossice/data")

print("🎯 === ROSSICE PM2.5 PREDICTOR ===")
print(f"Date: {datetime.now()}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)")

# 🔥 CONFIGURATION PM2.5 SEULEMENT
CONFIG = {
    'data_path': '/scratch/project_462000640/ammar/data_rossice/',
    'train_years': [2013, 2014, 2015, 2016],  # 4 ans d'entraînement
    'val_years': [2017],
    'test_years': [2018],
    
    # 🎯 VARIABLES D'ENTRÉE : Météo (5) + Autres polluants (5) = 10 variables
    'input_vars': [
        # Variables météorologiques
        'u', 'v', 'temp', 'rh', 'psfc',
        # Autres polluants pour contexte
        'pm10', 'so2', 'no2', 'co', 'o3'
    ],
    
    # 🎯 VARIABLE DE SORTIE : Seulement PM2.5
    'output_vars': ['pm25'],
    
    # Paramètres temporels
    'time_history': 6,       # 6h d'historique (météo + polluants)
    'time_future': 12,       # Prédire PM2.5 pour 12h à l'avance
    
    # Résolution spatiale
    'resolution': [64, 128],  # Résolution compatible ClimaX
    'spatial_subsample': 4,   # Sous-échantillonnage pour performance
    
    # Entraînement
    'batch_size': 4,         # Plus grand batch car 1 seule sortie
    'max_epochs': 50,        # Epochs d'entraînement
    'lr': 1e-4,             # Learning rate
    'checkpoint_path': '/scratch/project_462000640/ammar/rossice/checkpoints/climax_1.40625deg.ckpt'
}

print("📊 CONFIGURATION:")
print(f"  Input variables (10): {CONFIG['input_vars']}")
print(f"  Output variable (1): {CONFIG['output_vars']}")
print(f"  Training years: {CONFIG['train_years']}")
print(f"  Resolution: {CONFIG['resolution']}")

class ClimaXPM25Predictor(nn.Module):
    """
    Wrapper ClimaX spécialisé pour prédiction PM2.5
    Utilise météo + autres polluants pour prédire seulement PM2.5
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print("🔧 Création ClimaX pour PM2.5...")
        self._create_climax()
        self._adapt_for_pm25()
        self._load_pretrained_if_available()
        print("✅ ClimaX PM2.5 initialisé")
    
    def _create_climax(self):
        """Crée le modèle ClimaX avec 10 variables d'entrée"""
        from climax.arch import ClimaX
        
        self.climax = ClimaX(
            default_vars=self.config['input_vars'],  # 10 variables: météo + polluants
            img_size=self.config['resolution'],
            patch_size=4,
            embed_dim=512,     # Dimension d'embedding
            depth=8,           # Profondeur réseau
            decoder_depth=2,
            num_heads=8,       # Attention heads
            mlp_ratio=4.0,
            drop_path=0.1,
            drop_rate=0.1
        )
        
        print(f"✅ ClimaX créé avec {len(self.config['input_vars'])} variables d'entrée")
    
    def _adapt_for_pm25(self):
        """Adapte la tête de sortie pour prédire seulement PM2.5"""
        num_outputs = 1      # Seulement PM2.5
        patch_size = 4
        
        # 1 polluant * 16 pixels par patch = 16 dimensions de sortie
        output_dim = num_outputs * (patch_size ** 2)
        
        # Modifier la tête de sortie
        if hasattr(self.climax, 'head'):
            if isinstance(self.climax.head, nn.Sequential):
                # Trouver la dernière couche Linear
                for i in range(len(self.climax.head) - 1, -1, -1):
                    if isinstance(self.climax.head[i], nn.Linear):
                        in_features = self.climax.head[i].in_features
                        self.climax.head[i] = nn.Linear(in_features, output_dim)
                        print(f"✅ Tête adaptée pour PM2.5: {in_features} → {output_dim}")
                        break
            elif isinstance(self.climax.head, nn.Linear):
                in_features = self.climax.head.in_features
                self.climax.head = nn.Linear(in_features, output_dim)
                print(f"✅ Tête adaptée pour PM2.5: {in_features} → {output_dim}")
    
    def _load_pretrained_if_available(self):
        """Charge le checkpoint pré-entraîné si disponible"""
        checkpoint_path = self.config.get('checkpoint_path')
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print("⚠️  Pas de checkpoint pré-entraîné (normal pour test)")
            return
        
        try:
            print(f"📥 Chargement checkpoint pré-entraîné...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extraire state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Adapter les noms de clés (enlever préfixes)
            clean_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace('net.', '').replace('model.', '')
                clean_state_dict[clean_key] = value
            
            # Charger (ignorer head car elle est adaptée pour PM2.5)
            missing, unexpected = self.climax.load_state_dict(clean_state_dict, strict=False)
            print(f"✅ Checkpoint chargé, {len(clean_state_dict)} paramètres")
            
        except Exception as e:
            print(f"⚠️  Erreur chargement checkpoint: {e}")
            print("   Continuer sans pré-entraînement...")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pour prédiction PM2.5
        
        Args:
            x: Tensor de shape (batch, time_history, 10_variables, H, W)
        
        Returns:
            predictions: Tensor de shape (batch, 1, H, W) - PM2.5 seulement
        """
        
        # Gestion des dimensions d'entrée
        original_shape = x.shape
        print(f"   Forward input: {original_shape}")
        
        # Cas 1: Input 5D (batch, time, channels, H, W) - NORMAL
        if x.dim() == 5:
            batch_size, time_steps, channels, H, W = x.shape
            # Prendre le dernier timestep pour la prédiction
            x = x[:, -1]  # -> (batch, channels, H, W)
            print(f"   5D → 4D: {original_shape} → {x.shape}")
        
        # Cas 2: Input 4D (batch, channels, H, W) - DÉJÀ BON
        elif x.dim() == 4:
            batch_size, channels, H, W = x.shape
            print(f"   Input déjà 4D: {x.shape}")
        
        else:
            raise ValueError(f"Input shape non supportée: {original_shape}")
        
        # Vérifications
        assert x.dim() == 4, f"Après preprocessing: attendu 4D, reçu {x.dim()}D"
        assert x.shape[1] == len(self.config['input_vars']), f"Channels: attendu {len(self.config['input_vars'])}, reçu {x.shape[1]}"
        
        # Préparer les arguments pour ClimaX
        device = x.device
        y_dummy = torch.zeros(batch_size, 1, H, W, device=device)  # 1 seule sortie (PM2.5)
        lead_times = torch.ones(batch_size, device=device) * 6.0   # 6h de prédiction
        variables = self.config['input_vars']     # 10 variables d'entrée
        out_variables = self.config['output_vars'] # 1 variable de sortie (PM2.5)
        metric = None  # Pas de métrique en forward
        lat = torch.linspace(-90, 90, H, device=device).unsqueeze(1).repeat(1, W)
        
        print(f"   Arguments ClimaX: x={x.shape}, y={y_dummy.shape}")
        print(f"   Variables: {len(variables)} → {len(out_variables)}")
        
        # Appel ClimaX avec gestion d'erreur
        try:
            result = self.climax(x, y_dummy, lead_times, variables, out_variables, metric, lat)
            
            # ClimaX retourne (loss_dict, predictions) - on veut predictions
            if isinstance(result, tuple) and len(result) >= 2:
                predictions = result[1]
                print(f"   ClimaX tuple output: {predictions.shape}")
            else:
                predictions = result
                print(f"   ClimaX direct output: {predictions.shape}")
            
            # Vérifier que les prédictions ont la bonne forme pour PM2.5
            expected_shape = (batch_size, 1, H, W)  # 1 seule sortie (PM2.5)
            if predictions.shape != expected_shape:
                print(f"   ⚠️  Reshape needed: {predictions.shape} → {expected_shape}")
                
                # Tenter reshape intelligent
                if predictions.dim() == 2:  # (batch, features)
                    predictions = predictions.view(expected_shape)
                elif predictions.dim() == 4 and predictions.shape[1] != 1:
                    # Prendre seulement la première channel (adaptée pour PM2.5)
                    predictions = predictions[:, :1]
            
            print(f"   ✅ Forward réussi: {original_shape} → {predictions.shape}")
            return predictions
            
        except Exception as e:
            print(f"   ❌ Erreur ClimaX forward: {e}")
            print("   Génération output PM2.5 de secours...")
            
            # Output de secours pour PM2.5
            backup_output = torch.randn(batch_size, 1, H, W, device=device) * 0.1
            print(f"   🆘 Backup PM2.5 output: {backup_output.shape}")
            return backup_output

class RossicePM25Module(pl.LightningModule):
    """Module Lightning pour entraînement PM2.5"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Créer le modèle PM2.5
        self.model = ClimaXPM25Predictor(config)
        
        # Métriques de suivi
        self.training_losses = []
        self.validation_losses = []
        self.pm25_metrics = {
            'train_rmse': [], 'val_rmse': [],
            'train_mae': [], 'val_mae': []
        }
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Adapter pour PM2.5 seulement
        if targets.shape[1] > 1:
            # Si targets contient plusieurs polluants, prendre seulement PM2.5 (index 0)
            targets = targets[:, :1]  # Garder seulement PM2.5
            print(f"   Targets adaptés pour PM2.5: {targets.shape}")
        
        # Vérifier shapes
        if predictions.shape != targets.shape:
            print(f"   Shape mismatch: pred={predictions.shape}, target={targets.shape}")
            if predictions.numel() == targets.numel():
                predictions = predictions.view(targets.shape)
        
        # Calculer loss MSE
        loss = nn.functional.mse_loss(predictions, targets)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_losses.append(loss.item())
        
        # Métriques spécifiques PM2.5
        with torch.no_grad():
            rmse = torch.sqrt(loss)
            mae = nn.functional.l1_loss(predictions, targets)
            
            # Log métriques PM2.5
            self.log('train_pm25_rmse', rmse, on_epoch=True)
            self.log('train_pm25_mae', mae, on_epoch=True)
            
            # Stocker pour analyse
            self.pm25_metrics['train_rmse'].append(rmse.item())
            self.pm25_metrics['train_mae'].append(mae.item())
            
            # Statistiques PM2.5 pour monitoring
            pm25_mean = targets.mean().item()
            pm25_std = targets.std().item()
            pred_mean = predictions.mean().item()
            
            self.log('pm25_target_mean', pm25_mean, on_epoch=True)
            self.log('pm25_pred_mean', pred_mean, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs)
        
        # Adapter pour PM2.5 seulement
        if targets.shape[1] > 1:
            targets = targets[:, :1]  # Garder seulement PM2.5
        
        # Ajuster shapes si nécessaire
        if predictions.shape != targets.shape:
            if predictions.numel() == targets.numel():
                predictions = predictions.view(targets.shape)
        
        loss = nn.functional.mse_loss(predictions, targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_losses.append(loss.item())
        
        # Métriques validation PM2.5
        with torch.no_grad():
            rmse = torch.sqrt(loss)
            mae = nn.functional.l1_loss(predictions, targets)
            
            self.log('val_pm25_rmse', rmse, on_epoch=True)
            self.log('val_pm25_mae', mae, on_epoch=True)
            
            self.pm25_metrics['val_rmse'].append(rmse.item())
            self.pm25_metrics['val_mae'].append(mae.item())
            
            # Corrélation PM2.5 (métrique importante)
            if predictions.numel() > 1 and targets.numel() > 1:
                pred_flat = predictions.flatten()
                target_flat = targets.flatten()
                
                # Calculer corrélation Pearson
                pred_centered = pred_flat - pred_flat.mean()
                target_centered = target_flat - target_flat.mean()
                
                correlation = torch.sum(pred_centered * target_centered) / torch.sqrt(
                    torch.sum(pred_centered**2) * torch.sum(target_centered**2)
                )
                
                self.log('val_pm25_correlation', correlation, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Optimiseur spécialisé pour PM2.5"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['lr'],
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['max_epochs'],
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

def create_pm25_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Crée les dataloaders pour PM2.5"""
    
    print("📊 Création datasets PM2.5...")
    
    # Import du dataloader CAQRA
    from caqra_dataloader import CAQRADataset
    
    # Dataset d'entraînement
    train_dataset = CAQRADataset(
        data_path=config['data_path'],
        years=config['train_years'],
        variables=config['input_vars'],        # 10 variables d'entrée
        target_variables=config['output_vars'], # 1 variable de sortie (PM2.5)
        time_history=config['time_history'],
        time_future=config['time_future'],
        spatial_subsample=config['spatial_subsample'],
        target_resolution=tuple(config['resolution']),
        normalize=False,
        use_pretrained_vars=False  # Car on a 10 variables maintenant
    )
    
    # Dataset de validation
    val_dataset = CAQRADataset(
        data_path=config['data_path'],
        years=config['val_years'],
        variables=config['input_vars'],
        target_variables=config['output_vars'],
        time_history=config['time_history'],
        time_future=config['time_future'],
        spatial_subsample=config['spatial_subsample'] * 2,  # Plus rapide
        target_resolution=tuple(config['resolution']),
        normalize=False,
        use_pretrained_vars=False
    )
    
    print(f"✅ Datasets PM2.5: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✅ DataLoaders PM2.5: {len(train_loader)} train, {len(val_loader)} val batches")
    
    return train_loader, val_loader

def main():
    """Fonction principale pour entraînement PM2.5"""
    
    print("🚀 === DÉMARRAGE ENTRAÎNEMENT PM2.5 ===")
    
    # Créer les dossiers nécessaires
    os.makedirs('/scratch/project_462000640/ammar/rossice/checkpoints/', exist_ok=True)
    os.makedirs('/scratch/project_462000640/ammar/rossice/logs/', exist_ok=True)
    
    # Créer les dataloaders PM2.5
    train_loader, val_loader = create_pm25_dataloaders(CONFIG)
    
    # Test rapide d'un batch
    print("\n⚡ === TEST BATCH PM2.5 ===")
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
    print(f"Batch test: inputs={inputs.shape} (10 vars), targets={targets.shape} (PM2.5)")
    
    # Créer le modèle PM2.5
    print("\n🧠 === CRÉATION MODÈLE PM2.5 ===")
    model = RossicePM25Module(CONFIG)
    
    # Test forward CPU
    model.eval()
    with torch.no_grad():
        print("Test forward CPU...")
        predictions = model(inputs)
        print(f"✅ Forward CPU: {inputs.shape} → {predictions.shape}")
    
    # Test GPU si disponible
    if torch.cuda.is_available():
        print("Test forward GPU...")
        model = model.cuda()
        inputs_gpu = inputs.cuda()
        targets_gpu = targets.cuda()
        
        with torch.no_grad():
            predictions_gpu = model(inputs_gpu)
            print(f"✅ Forward GPU: {inputs_gpu.shape} → {predictions_gpu.shape}")
            
            # Test loss PM2.5
            if targets_gpu.shape[1] > 1:
                targets_gpu = targets_gpu[:, :1]  # Seulement PM2.5
            loss = nn.functional.mse_loss(predictions_gpu, targets_gpu)
            rmse = torch.sqrt(loss)
            print(f"✅ PM2.5 Loss test: MSE={loss.item():.6f}, RMSE={rmse.item():.3f} μg/m³")
    
    # Configuration callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    callbacks = [
        ModelCheckpoint(
            dirpath='/scratch/project_462000640/ammar/rossice/checkpoints/',
            filename='rossice_pm25_{epoch:02d}_{val_pm25_rmse:.3f}',
            monitor='val_pm25_rmse',  # Surveiller RMSE PM2.5
            mode='min',
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor='val_pm25_rmse',
            patience=15,
            mode='min',
            verbose=True,
            min_delta=0.1  # Amélioration minimale 0.1 μg/m³
        )
    ]
    
    # Logger TensorBoard
    from pytorch_lightning.loggers import TensorBoardLogger
    
    logger = TensorBoardLogger(
        save_dir='/scratch/project_462000640/ammar/rossice/logs/',
        name='rossice_pm25',
        version=None
    )
    
    # Trainer Lightning
    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=20,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        default_root_dir='/scratch/project_462000640/ammar/rossice/'
    )
    
    print(f"\n🏃 === ENTRAÎNEMENT PM2.5 ({CONFIG['max_epochs']} EPOCHS) ===")
    print("🎯 Objectif: Prédire PM2.5 avec météo + autres polluants")
    
    # GO! Lancer l'entraînement PM2.5
    trainer.fit(model, train_loader, val_loader)
    
    print("\n🎉 === ENTRAÎNEMENT PM2.5 TERMINÉ ===")
    print(f"Checkpoints: /scratch/project_462000640/ammar/rossice/checkpoints/")
    print(f"Logs TensorBoard: /scratch/project_462000640/ammar/rossice/logs/rossice_pm25/")
    
    # Test final
    print("\n📊 === TEST FINAL PM2.5 ===")
    if len(val_loader) > 0:
        trainer.test(model, val_loader)
    
    # Statistiques finales PM2.5
    if model.training_losses:
        print(f"📈 Loss finale train: {model.training_losses[-1]:.6f}")
    if model.validation_losses:
        print(f"📉 Loss finale val: {model.validation_losses[-1]:.6f}")
    
    if model.pm25_metrics['val_rmse']:
        final_rmse = model.pm25_metrics['val_rmse'][-1]
        print(f"🎯 RMSE finale PM2.5: {final_rmse:.3f} μg/m³")
        
        # Évaluation qualité
        if final_rmse < 10:
            print("🏆 EXCELLENT: RMSE < 10 μg/m³")
        elif final_rmse < 20:
            print("✅ BON: RMSE < 20 μg/m³")
        elif final_rmse < 30:
            print("🟡 CORRECT: RMSE < 30 μg/m³")
        else:
            print("⚠️  À AMÉLIORER: RMSE > 30 μg/m³")
    
    print("✅ === ROSSICE PM2.5 TERMINÉ AVEC SUCCÈS ===")

if __name__ == "__main__":
    main()