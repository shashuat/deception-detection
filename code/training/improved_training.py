# training/improved_training.py

import torch
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler

def mixup_data(x, y, alpha=0.2, device='cpu'):
    """
    Applies mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Applies the mixup loss calculation
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_with_improvements(config, train_loader, model, optimizer, criterion, 
                           sub_loss, sub_labels_loss, scheduler=None, 
                           use_mixup=True, mixup_alpha=0.2,
                           use_amp=True, accumulation_steps=4):
    """
    Enhanced training function with:
    1. Gradient accumulation for larger effective batch size
    2. Mixed precision training for speed
    3. Mixup augmentation for regularization
    4. Improved progress tracking
    
    Args:
        config: Training configuration
        train_loader: DataLoader for training data
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        sub_loss: Dictionary of loss functions for sub-tasks
        sub_labels_loss: List of loss functions for sub-labels
        scheduler: Learning rate scheduler
        use_mixup: Whether to use mixup augmentation
        mixup_alpha: Alpha parameter for mixup
        use_amp: Whether to use automatic mixed precision
        accumulation_steps: Number of steps to accumulate gradients
    """
    device = config["device"]
    model.train()
    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    epoch_subloss = []
    epoch_subpreds = {}
    epoch_sub_labels_loss = []
    
    # For mixed precision training
    scaler = GradScaler() if use_amp else None
    
    # Set for gradient accumulation
    optimizer.zero_grad()
    
    for i, (train_data, labels, sub_labels) in enumerate(train_loader):
        # Determine if this is an accumulation step
        is_accumulation_step = (i + 1) % accumulation_steps != 0
        
        # Move data to device
        labels = labels.to(device)
        sub_labels = sub_labels.to(device)
        model_input = {k: v.to(device) for k, v in train_data.items()}
        
        # Apply mixup if enabled
        if use_mixup and random.random() < 0.8:  # Apply mixup 80% of the time
            # For simplicity, we'll just mixup the audio modality if it exists
            if 'audio' in model_input:
                model_input['audio'], labels_a, labels_b, lam = mixup_data(
                    model_input['audio'], labels, mixup_alpha, device
                )
                use_mixup_loss = True
            else:
                use_mixup_loss = False
        else:
            use_mixup_loss = False
        
        # Forward pass with mixed precision
        with autocast() if use_amp else torch.no_grad():
            outputs, outputs_sub_labels, sub_outputs = model(model_input)
            
            # Calculate loss
            if use_mixup_loss:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Multi-task losses
            if config["multi"] and sub_outputs is not None:
                secondary_loss = {}
                for k, crit in sub_loss.items():
                    if use_mixup_loss:
                        secondary_loss[k] = mixup_criterion(
                            crit, sub_outputs[k], labels_a, labels_b, lam
                        )
                    else:
                        secondary_loss[k] = crit(sub_outputs[k], labels)
                
                loss = 0.7 * loss + 0.3 * sum(secondary_loss.values()) / len(secondary_loss)
                epoch_subloss.append({k: v.item() for k, v in secondary_loss.items()})
                
                for k, v in sub_outputs.items():
                    epoch_subpreds[k] = epoch_subpreds.get(k, []) + [torch.argmax(v, dim=1)]
            
            # Sub-label losses
            if config["sub_labels"]:
                sub_labels_loss_values = []
                for idx in range(len(sub_labels_loss)):
                    sub_labels_loss_values.append(
                        sub_labels_loss[idx](outputs_sub_labels[idx], sub_labels[:, idx])
                    )
                
                loss = 0.85 * loss + 0.15 * sum(sub_labels_loss_values) / len(sub_labels_loss_values)
                epoch_sub_labels_loss.append([v.item() for v in sub_labels_loss_values])
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass with mixed precision
        if use_amp:
            scaler.scale(loss).backward()
            if not is_accumulation_step:
                # Apply gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            loss.backward()
            if not is_accumulation_step:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        
        # Track progress - multiply by accumulation_steps to get the actual loss
        epoch_loss.append(loss.item() * accumulation_steps)
        epoch_preds.append(torch.argmax(outputs, dim=1))
        epoch_labels.append(labels)
        
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item() * accumulation_steps:.5f}")
    
    # Combine results
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)
    avg_loss = np.mean(epoch_loss)
    
    # Handle multi-task results
    avg_sub_loss = {}
    if config["multi"]:
        for info in epoch_subloss:
            for k, v in info.items():
                avg_sub_loss[k] = avg_sub_loss.get(k, []) + [v]
        
        for k in avg_sub_loss:
            avg_sub_loss[k] = sum(avg_sub_loss[k]) / len(avg_sub_loss[k])
            epoch_subpreds[k] = torch.cat(epoch_subpreds[k])
    
    # Process sub-label losses
    epoch_sub_labels_loss_tensor = torch.tensor(epoch_sub_labels_loss) if epoch_sub_labels_loss else torch.tensor([])
    mean_loss_per_head = epoch_sub_labels_loss_tensor.mean(dim=0) if epoch_sub_labels_loss_tensor.numel() > 0 else torch.tensor([])
    mean_loss_per_head_list = mean_loss_per_head.tolist() if mean_loss_per_head.numel() > 0 else []
    
    return avg_loss, epoch_preds, epoch_labels, avg_sub_loss, epoch_subpreds, mean_loss_per_head_list


def validate_with_improvements(config, val_loader, model, criterion, sub_loss, use_amp=True):
    """Enhanced validation function with mixed precision support"""
    model.eval()
    device = config["device"]
    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    epoch_subloss = []
    epoch_subpreds = {}
    
    with torch.no_grad():
        for val_data, labels, sub_labels in val_loader:
            # Move data to device
            labels = labels.to(device)
            model_input = {k: v.to(device) for k, v in val_data.items()}
            
            # Forward pass with mixed precision
            with autocast() if use_amp else torch.no_grad():
                outputs, outputs_sub_labels, sub_outputs = model(model_input)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Multi-task losses
                if config["multi"] and sub_outputs is not None:
                    secondary_loss = {
                        k: criterion(sub_outputs[k], labels)
                        for k, criterion in sub_loss.items()
                    }
                    loss = 0.7 * loss + 0.3 * sum(secondary_loss.values()) / len(secondary_loss)
                    epoch_subloss.append({k: v.item() for k, v in secondary_loss.items()})
                    
                    for k, v in sub_outputs.items():
                        epoch_subpreds[k] = epoch_subpreds.get(k, []) + [torch.argmax(v, dim=1)]
            
            # Track progress
            epoch_loss.append(loss.item())
            epoch_preds.append(torch.argmax(outputs, dim=1))
            epoch_labels.append(labels)
    
    # Combine results
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)
    avg_loss = np.mean(epoch_loss)
    
    # Handle multi-task results
    avg_sub_loss = {}
    if config["multi"] and epoch_subloss:
        for info in epoch_subloss:
            for k, v in info.items():
                avg_sub_loss[k] = avg_sub_loss.get(k, []) + [v]
        
        for k in avg_sub_loss:
            avg_sub_loss[k] = sum(avg_sub_loss[k]) / len(avg_sub_loss[k])
            epoch_subpreds[k] = torch.cat(epoch_subpreds[k])
    
    return avg_loss, epoch_preds, epoch_labels, avg_sub_loss, epoch_subpreds