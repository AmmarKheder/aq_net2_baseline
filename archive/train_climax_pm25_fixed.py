#!/usr/bin/env python3
"""
ClimaX fine-tuning for PM2.5 - Train/Val/Test split + Metrics + Early Stopping
"""
import os
import argparse
from pathlib import Path
import time

# ─── MIOpen cache fix ─────────────────────────────────────────────────────────
print("🔧 Configuration système...")
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/scratch/project_462000640/ammar/miopen_cache"
os.environ["MIOPEN_USER_DB_PATH"]      = "/scratch/project_462000640/ammar/miopen_cache"
os.environ["MIOPEN_DISABLE_CACHE"]     = "0"
os.makedirs(os.environ["MIOPEN_CUSTOM_CACHE_DIR"], exist_ok=True)
os.environ["TIMM_FUSED_ATTN"] = "0"
print("✅ Configuration MIOpen terminée")

print("📦 Chargement des modules...")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, '/scratch/project_462000640/ammar/rossice/climax/src')
sys.path.insert(0, '/scratch/project_462000640/ammar/rossice/data')
from climax.arch import ClimaX
from caqra_dataloader import CAQRADataset
print("✅ Modules chargés avec succès")

# ─── Model (head-only fine-tuning) ───────────────────────────────────────────
class PM25Model(nn.Module):
    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()
        print(f"🏗️  Initialisation du modèle PM25Model...")
        print(f"   📍 Device: {device}")
        print(f"   📄 Checkpoint: {checkpoint_path}")
        
        # Variables météo + pollution
        self.variables = ['u','v','temp','rh','psfc','pm10','so2','no2','co','o3']
        print(f"   🌡️  Variables d'entrée: {self.variables}")
        
        # Backbone ClimaX
        print("   🧠 Construction du backbone ClimaX...")
        self.climax = ClimaX(
            default_vars=self.variables,
            img_size=[128,256],
            patch_size=4,
            embed_dim=1024,
            depth=8, decoder_depth=2,
            num_heads=16, mlp_ratio=4,
            drop_path=0.1, drop_rate=0.1
        )
        print("   ✅ Backbone ClimaX créé")
        
        # Charger les poids (sauf la head)
        print("   📥 Chargement des poids pré-entraînés...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        sd = {k[4:]:v for k,v in ckpt.items() if k.startswith('net.')}
        sd = {k:v for k,v in sd.items() if not k.startswith('head.')}
        self.climax.load_state_dict(sd, strict=False)
        print(f"   ✅ {len(sd)} poids chargés depuis le checkpoint")
        
        # Geler l'encoder
        frozen_params = 0
        for p in self.climax.parameters():
            p.requires_grad = False
            frozen_params += p.numel()
        print(f"   🧊 Encoder gelé ({frozen_params:,} paramètres)")
        
        # Tête de régression PM2.5
        embed_dim = 1024
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 1, 1)
        )
        trainable_params = sum(p.numel() for p in self.head.parameters())
        print(f"   🎯 Tête de régression créée ({trainable_params:,} paramètres entraînables)")
        
        self.to(device)
        print("✅ Modèle PM25Model initialisé avec succès")

    def forward(self, x):
        B,_,H,W = x.shape
        device = x.device
        lt = torch.zeros(B, device=device)
        feats = self.climax.forward_encoder(x, lt, self.variables)
        for blk in self.climax.blocks:
            feats = blk(feats)
        p = self.climax.patch_size
        h, w = H // p, W // p
        num = h * w
        feats = feats[:, :num]\
            .reshape(B, h, w, -1)\
            .permute(0, 3, 1, 2)
        if feats.shape[-2:] != (H, W):
            feats = nn.functional.interpolate(
                feats, size=(H, W), mode='bilinear', align_corners=False
            )
        return self.head(feats).squeeze(1)

# ─── Évaluation ───────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    print("   🔍 Évaluation en cours...")
    model.eval()
    all_pred, all_true = [], []
    eval_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X[:, -1].to(device)
            y = y.squeeze(1).to(device)
            pred = model(X)
            all_pred.append(pred.cpu().numpy().ravel())
            all_true.append(y.cpu().numpy().ravel())
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(f"      Batch {batch_idx}/{len(loader)} évalué...")
    
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    eval_time = time.time() - eval_start
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':   r2_score(y_true, y_pred)
    }
    
    print(f"   ✅ Évaluation terminée en {eval_time:.2f}s ({len(y_pred):,} prédictions)")
    return metrics

# ─── Script principal ─────────────────────────────────────────────────────────
def main(args):
    print("🚀 Démarrage du fine-tuning ClimaX pour PM2.5")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device utilisé: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Modèle, optimiseur, scheduler, loss
    print("\n🏗️  Construction du modèle...")
    model = PM25Model(args.checkpoint, device=device)
    
    print("\n⚙️  Configuration de l'optimisation...")
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    print(f"   Optimiseur: Adam (lr={args.lr})")
    print(f"   Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"   Loss: MSE")

    # Jeu d'entraînement
    print(f"\n📚 Chargement des données d'entraînement ({args.train_years})...")
    train_ds = CAQRADataset(
        data_path=args.data,
        years=args.train_years,
        variables=args.variables,
        target_variables=['pm25'],
        time_history=4,
        time_future=1,
        normalize=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    print(f"✅ Dataset d'entraînement: {len(train_ds):,} échantillons, {len(train_loader)} batches")

    # Jeu de validation
    print(f"\n📖 Chargement des données de validation ({args.val_years})...")
    val_ds = CAQRADataset(
        data_path=args.data,
        years=args.val_years,
        variables=args.variables,
        target_variables=['pm25'],
        time_history=4,
        time_future=1,
        normalize=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    print(f"✅ Dataset de validation: {len(val_ds):,} échantillons, {len(val_loader)} batches")

    # Jeu de test
    print(f"\n📋 Chargement des données de test ({args.test_years})...")
    test_ds = CAQRADataset(
        data_path=args.data,
        years=args.test_years,
        variables=args.variables,
        target_variables=['pm25'],
        time_history=4,
        time_future=1,
        normalize=True
    )
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    print(f"✅ Dataset de test: {len(test_ds):,} échantillons, {len(test_loader)} batches")

    # Résumé des paramètres
    print(f"\n📊 Résumé de l'entraînement:")
    print(f"   Époques max: {args.epochs}")
    print(f"   Batch size: {args.bs}")
    print(f"   Early stopping patience: {args.patience}")
    print(f"   Variables: {len(args.variables)} ({', '.join(args.variables)})")

    print("\n" + "=" * 60)
    print("🎯 DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 60)

    best_rmse, no_improve = float('inf'), 0
    total_train_time = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n📈 Époque {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Phase entraînement
        print("🔥 Phase d'entraînement...")
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X[:, -1].to(device)
            y = y.squeeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / num_batches
        print(f"✅ Entraînement terminé - Loss moyenne: {avg_train_loss:.6f}")

        # Validation
        print("🔍 Phase de validation...")
        metrics = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        
        print(f"📊 Résultats époque {epoch}:")
        print(f"   Train Loss: {avg_train_loss:.6f}")
        print(f"   Val RMSE:   {metrics['rmse']:.4f}")
        print(f"   Val MAE:    {metrics['mae']:.4f}")
        print(f"   Val R²:     {metrics['r2']:.4f}")
        print(f"   Temps:      {epoch_time:.1f}s")
        print(f"   LR actuel:  {optimizer.param_groups[0]['lr']:.2e}")

        # Scheduler & early stopping
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(metrics['rmse'])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f"📉 Learning rate réduit: {old_lr:.2e} → {new_lr:.2e}")
        
        if metrics['rmse'] < best_rmse:
            best_rmse, no_improve = metrics['rmse'], 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            print(f"🏆 Nouveau meilleur modèle sauvé! RMSE: {best_rmse:.4f}")
        else:
            no_improve += 1
            print(f"⏸️  Pas d'amélioration ({no_improve}/{args.patience})")
            
            if no_improve >= args.patience:
                print(f"\n⏱️  Early stopping déclenché après {epoch} époques")
                print(f"   Meilleur RMSE: {best_rmse:.4f}")
                break

    print("\n" + "=" * 60)
    print("🏁 ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    print(f"⏱️  Temps total d'entraînement: {total_train_time/60:.1f} minutes")
    print(f"🏆 Meilleur RMSE validation: {best_rmse:.4f}")

    # Test final
    print(f"\n🧪 Évaluation finale sur le jeu de test...")
    print("-" * 40)
    
    # Charger le meilleur modèle
    if os.path.exists('checkpoints/best_model.pt'):
        print("📥 Chargement du meilleur modèle...")
        model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n🎯 RÉSULTATS FINAUX:")
    print("=" * 30)
    print(f"Test MAE:  {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    print("=" * 30)
    
    print(f"\n✅ Fine-tuning terminé avec succès!")
    print(f"💾 Modèle sauvé dans: checkpoints/best_model.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str,   default="/scratch/project_462000640/ammar/data_rossice/")
    parser.add_argument('--checkpoint', type=str,   default="checkpoints/climax_1.40625deg.ckpt")
    parser.add_argument('--train_years',type=int,   nargs='+', default=[2013,2014,2015])
    parser.add_argument('--val_years',  type=int,   nargs='+', default=[2016,2017])
    parser.add_argument('--test_years', type=int,   nargs='+', default=[2018])
    parser.add_argument('--variables',  type=str,   nargs='+', default=['u','v','temp','rh','psfc','pm10','so2','no2','co','o3'])
    parser.add_argument('--bs',         type=int,   default=32)
    parser.add_argument('--epochs',     type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--patience',   type=int,   default=5)
    args = parser.parse_args()

    Path("checkpoints").mkdir(exist_ok=True)
    main(args)