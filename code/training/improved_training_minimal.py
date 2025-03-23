import torch
import time
# Small improvements to the training function with minimal changes
def train_one_epoch_improved(config, train_loader, model, optimizer, criterion, sub_loss, sub_labels_loss, scheduler=None):
    """Train model for one epoch with minimal but effective improvements"""
    model.train()
    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    epoch_subloss = []
    epoch_subpreds = {}
    epoch_sub_labels_loss = []
    start_time = time.time()
    grad_clip_value = 1.0  # Add gradient clipping to prevent unstable training
    
    for i, (train_data, labels, sub_labels) in enumerate(train_loader):
        # Prepare input
        labels = labels.to(config["device"])
        sub_labels = sub_labels.to(config["device"])
        model_input = {k: v.to(config["device"]) for k, v in train_data.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs, outputs_sub_labels, sub_outputs = model(model_input)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        if config["multi"] and sub_outputs is not None:
            secondary_loss = {
                k: criterion(sub_outputs[k], labels)
                for k, criterion in sub_loss.items()
            }
            loss = 0.7 * loss + 0.3 * sum(secondary_loss.values()) / len(secondary_loss)
            epoch_subloss.append({k: v.item() for k, v in secondary_loss.items()})

            for k, v in sub_outputs.items():
                epoch_subpreds[k] = epoch_subpreds.get(k, []) + [torch.argmax(v, dim=1)]

        if config["sub_labels"]:
            sub_labels_loss_values = []
            for idx in range(len(sub_labels_loss)):
                sub_labels_loss_values.append(sub_labels_loss[idx](outputs_sub_labels[idx], sub_labels[:, idx]))
            
            loss = 0.85 * loss + 0.15 * sum(sub_labels_loss_values) / len(sub_labels_loss_values)
            epoch_sub_labels_loss.append([v.item() for v in sub_labels_loss_values])
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping to prevent large updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate scheduler after each batch (if provided)
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Track progress
        epoch_loss.append(loss.item())
        epoch_preds.append(torch.argmax(outputs, dim=1))
        epoch_labels.append(labels)
        
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.5f}")
    
    # Combine results
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)
    avg_loss = np.mean(epoch_loss)
    avg_sub_loss = {}
    if config["multi"]:
        avg_sub_loss = {}
        for info in epoch_subloss:
            for k, v in info.items():
                avg_sub_loss[k] = avg_sub_loss.get(k, []) + [v]
        
        for k in avg_sub_loss:
            avg_sub_loss[k] = sum(avg_sub_loss[k]) / len(avg_sub_loss[k])
            epoch_subpreds[k] = torch.cat(epoch_subpreds[k])
    
    time_taken = time.time() - start_time
    print(f"Training completed in {time_taken:.2f} seconds")

    epoch_sub_labels_loss_tensor = torch.tensor(epoch_sub_labels_loss) if epoch_sub_labels_loss else torch.tensor([])
    mean_loss_per_head = epoch_sub_labels_loss_tensor.mean(dim=0) if epoch_sub_labels_loss_tensor.numel() > 0 else torch.tensor([])
    mean_loss_per_head_list = mean_loss_per_head.tolist() if mean_loss_per_head.numel() > 0 else []
    
    return avg_loss, epoch_preds, epoch_labels, avg_sub_loss, epoch_subpreds, mean_loss_per_head_list