import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def evaluate(model, loader, device):
    print("Evaluating...")
    model.eval()
    all_pred, all_true = [], []
    eval_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (X, y, lead_times) in enumerate(loader):
            X = X.to(device)  # Shape should be (B, C, H, W)
            if len(X.shape) == 3:
                X = X.unsqueeze(1)  # Add channel dim if missing  # Use only last timestep
            y = y.to(device)
            lead_times = torch.tensor(lead_times).to(device)
            pred = model(X, lead_times)
            all_pred.append(pred.cpu().numpy().ravel())
            all_true.append(y.cpu().numpy().ravel())
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(f"Batch {batch_idx}/{len(loader)} evaluated...")
    
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    eval_time = time.time() - eval_start
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':   r2_score(y_true, y_pred)
    }
    
    print(f"Evaluation finished in {eval_time:.2f}s ({len(y_pred):,} predictions)")
    return metrics

def train(config, model, train_loader, val_loader, device):
    print("Starting training...")
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_rmse, no_improve = float('inf'), 0
    total_train_time = 0

    for epoch in range(1, config['train']['epochs'] + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{config['train']['epochs']}")
        
        # Training phase
        print("Training phase...")
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X, y, lead_times) in enumerate(train_loader):
            X = X.to(device)  # Shape should be (B, C, H, W)
            if len(X.shape) == 3:
                X = X.unsqueeze(1)  # Add channel dim if missing  # Use only last timestep
            y = y.to(device)
            lead_times = torch.tensor(lead_times).to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(X, lead_times), y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / num_batches
        print(f"Training finished - Average loss: {avg_train_loss:.6f}")

        # Validation phase
        print("Validation phase...")
        metrics = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        
        print(f"Epoch {epoch} results:")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val RMSE:   {metrics['rmse']:.4f}")
        print(f"Val MAE:    {metrics['mae']:.4f}")
        print(f"Val R²:     {metrics['r2']:.4f}")
        print(f"Time:       {epoch_time:.1f}s")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Scheduler & early stopping
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(metrics['rmse'])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        if metrics['rmse'] < best_rmse:
            best_rmse, no_improve = metrics['rmse'], 0
            torch.save(model.state_dict(), 'outputs/checkpoints/best_model.pt')
            print(f"New best model saved! RMSE: {best_rmse:.4f}")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{config['train']['patience']})")
            
            if no_improve >= config['train']['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best RMSE: {best_rmse:.4f}")
                break

    print("\nTraining finished.")
    print(f"Total training time: {total_train_time/60:.1f} minutes")
    print(f"Best validation RMSE: {best_rmse:.4f}")
    return best_rmse

