# train_test.py with minimal improvements
import os
import sys
import time
import json
import traceback
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
import wandb

from dataloader.audio_visual_dataset import DatasetDOLOS, af_collate_fn
from models.audio_model import W2V2_Model
from models.fusion_model import Fusion
from models.visual_model import ViT_model

# Import the improved training function
from training.improved_training_minimal import train_one_epoch_improved

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from setup import get_env

ENV = get_env()
config = {
    # Paths
    "data_root": ENV["DOLOS_PATH"],
    "audio_path": ENV["AUDIO_PATH"],
    "visual_path": ENV["VISUAL_PATH"],
    "transcripts_path": ENV["TRANSCRIPTS_PATH"],
    "log_dir": ENV["LOGS_PATH"],
    
    # Training parameters - keep your existing values
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr": 1e-4,  # Keep your original learning rate
    "batch_size": 4,  # Keep your original batch size
    "num_epochs": 20,  # Keep your original number of epochs
    
    # Model configuration - keep your existing values
    "modalities": ["faces", "audio", "text", "whisper"],
    "num_layers": 4,
    "adapter": True,
    "adapter_type": "efficient_conv",  
    "fusion_type": "cross_attention",  
    "multi": True,  
    "sub_labels": False,
    
    # NEW: Small improvements
    "use_gradient_clipping": True,  # Add gradient clipping
    "weight_decay": 1e-5,  # Add small weight decay
    "scheduler_patience": 2,  # Reduce scheduler patience
    
    # Protocols for training and testing
    "protocols": [
        ["train_fold3.csv", "test_fold3.csv"],
        # Other protocols can be uncommented as needed
    ],
    
    # wandb configuration
    "wandb": {
        "project": "multimodal-lie-detection",
        "entity": None,
        "tags": ["multimodal", "lie-detection", "deep-learning", "improved"]
    }
}

# Keep the original validation function
def validate(config, val_loader, model, criterion, sub_loss):
    """Validate model on validation set"""
    model.eval()
    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    epoch_subloss = []
    epoch_subpreds = {}
    
    with torch.no_grad():
        for val_data, labels, sub_labels in val_loader:
            labels = labels.to(config["device"])
            model_input = {k: v.to(config["device"]) for k, v in val_data.items()}
            
            outputs, outputs_sub_labels, sub_outputs = model(model_input)
            
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
            
            epoch_loss.append(loss.item())
            epoch_preds.append(torch.argmax(outputs, dim=1))
            epoch_labels.append(labels)
    
    # Combine results
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)
    avg_loss = np.mean(epoch_loss)
    
    avg_sub_loss = {}
    if config["multi"]:
        for info in epoch_subloss:
            for k, v in info.items():
                avg_sub_loss[k] = avg_sub_loss.get(k, []) + [v]
        
        for k in avg_sub_loss:
            avg_sub_loss[k] = sum(avg_sub_loss[k]) / len(avg_sub_loss[k])
            epoch_subpreds[k] = torch.cat(epoch_subpreds[k])

    return avg_loss, epoch_preds, epoch_labels, avg_sub_loss, epoch_subpreds

def evaluate_metrics(labels, preds):
    """Calculate evaluation metrics"""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)  # Added zero_division=0 to handle warnings
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score

def train_and_evaluate():
    """Main training and evaluation function with minimal improvements"""
    # Initialize wandb
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        tags=config["wandb"]["tags"],
        config={k: v for k, v in config.items() if k != "wandb"},
        settings=wandb.Settings(start_method="thread")
    )
    
    # Create log directory
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Create model name
    model_name = f"IMPROVED_DOLOS_layers_{config['num_layers']}_Adapter_{config['adapter']}"
    if config["adapter"]:
        model_name += f"_type_{config['adapter_type']}"
    if config["multi"]:
        model_name += "_multi"
    
    # Set wandb run name
    wandb.run.name = model_name
        
    # Create log file
    log_file = os.path.join(config["log_dir"], f"{int(time.time())}_{model_name}.json")
    log_json = {
        "modalities": config["modalities"],
        "learning_rate": config["lr"],
        "batch_size": config["batch_size"],
        "num_epochs": config["num_epochs"],
        "num_layers": config["num_layers"],
        "adapter": config["adapter"],
        "runs": {}
    }

    if config["adapter"]:
        log_json["adapter_type"] = config["adapter_type"]

    with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)
    
    # Train and evaluate on each protocol
    try:
        for protocol in config["protocols"]:
            train_file, test_file = protocol
            protocol_key = f"protocol: [train] {train_file} | [test] {test_file}"
            
            print(f"\n\nRunning protocol: {train_file} (train), {test_file} (test)")
            wandb.config.update({"protocol_train": train_file, "protocol_test": test_file})
            
            if protocol_key in log_json["runs"]:
                raise Exception("You can run a protocol only one time.\nProtocol <" + protocol_key + "> was already run")
            
            log_json["runs"][protocol_key] = {
                "steps": [],
                "train_file": train_file,
                "test_file": test_file
            }
            with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)
            
            # Get annotation file paths
            train_anno = os.path.join(config["data_root"], "protocols", train_file)
            test_anno = os.path.join(config["data_root"], "protocols", test_file)
            
            # Create datasets
            train_dataset = DatasetDOLOS(
                annotations_file=train_anno, 
                audio_dir=config["audio_path"], 
                img_dir=config["visual_path"],
                transcripts_dir=config["transcripts_path"],
                ignore_audio_errors=True
            )

            test_dataset = DatasetDOLOS(
                annotations_file=test_anno, 
                audio_dir=config["audio_path"], 
                img_dir=config["visual_path"],
                transcripts_dir=config["transcripts_path"],
                ignore_audio_errors=True
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config["batch_size"], 
                shuffle=True,
                collate_fn=af_collate_fn,
                num_workers=4
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config["batch_size"], 
                shuffle=False,
                collate_fn=af_collate_fn,
                num_workers=4
            )
            
            print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
            wandb.log({
                "dataset/train_samples": len(train_dataset),
                "dataset/test_samples": len(test_dataset)
            })
            
            # Create model - keep your original model
            model = Fusion(
                config["fusion_type"],
                config["modalities"],
                config["num_layers"], 
                config["adapter"], 
                config["adapter_type"], 
                config["multi"],
                train_dataset.num_sub_labels
            )
            
            # Move model to device
            model.to_device(config["device"])
            print(f"Model created: {model_name}")
            
            # Create optimizer and loss function
            # IMPROVEMENT: Use AdamW with weight decay
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config["lr"],
                weight_decay=config["weight_decay"]
            )
            
            # IMPROVEMENT: Reduced patience for scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=config["scheduler_patience"]
            )
            
            criterion = nn.CrossEntropyLoss()
            
            # Additional loss functions for multitask learning
            secondary_loss = {}
            if config["multi"]:
                secondary_loss = {
                    k: nn.CrossEntropyLoss()
                    for k in config["modalities"]
                }
                print("Multitask learning enabled")

            sub_labels_loss = []
            if config["sub_labels"]:
                print(train_dataset.num_sub_labels)
                for _ in range(train_dataset.num_sub_labels):
                    sub_labels_loss.append(nn.CrossEntropyLoss())
            
            # Training loop
            best_acc = 0.0
            best_f1 = 0.0  # IMPROVEMENT: Track F1 score too
            early_stop_counter = 0  # IMPROVEMENT: Add early stopping
            early_stop_patience = 5
            print(f"Starting training for {config['num_epochs']} epochs")
            
            for epoch in range(config["num_epochs"]):
                print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
                
                # IMPROVEMENT: Use the improved training function
                train_loss, train_preds, train_labels, train_sub_loss, train_sub_preds, sub_labels_loss_values = train_one_epoch_improved(
                    config, train_loader, model, optimizer, criterion, secondary_loss, sub_labels_loss
                )
                
                # Calculate training metrics
                train_acc, train_f1, train_auc = evaluate_metrics(
                    train_labels.cpu().numpy(), train_preds.cpu().numpy()
                )

                if config["multi"]:
                    for k in train_sub_preds:
                        train_sub_preds[k] = evaluate_metrics(
                            train_labels.cpu().numpy(), train_sub_preds[k].cpu().numpy()
                        )
                
                # Validate
                val_loss, val_preds, val_labels, val_sub_loss, val_sub_preds = validate(
                    config, test_loader, model, criterion, secondary_loss
                )
                
                # Calculate validation metrics
                val_acc, val_f1, val_auc = evaluate_metrics(
                    val_labels.cpu().numpy(), val_preds.cpu().numpy()
                )

                if config["multi"]:
                    for k in val_sub_preds:
                        val_sub_preds[k] = evaluate_metrics(
                            val_labels.cpu().numpy(), val_sub_preds[k].cpu().numpy()
                        )
                
                # Log metrics to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/f1": train_f1,
                    "train/auc": train_auc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/f1": val_f1,
                    "val/auc": val_auc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                # Print results
                print(f"Epoch {epoch+1} Results:")
                print(f"  Train - Loss: {train_loss:.5f}, Acc: {train_acc:.5f}, F1: {train_f1:.5f}, AUC: {train_auc:.5f}")
                print(f"  Valid - Loss: {val_loss:.5f}, Acc: {val_acc:.5f}, F1: {val_f1:.5f}, AUC: {val_auc:.5f}")
                print(f"  Train - Sub-Loss: {train_sub_loss} | {train_sub_preds}")
                print(f"  Valid - Sub-Loss: {val_sub_loss} | {val_sub_preds}")
                
                # Update scheduler with validation accuracy (same as original)
                scheduler.step(val_acc)
                
                # Update log file
                log_json["runs"][protocol_key]["steps"].append({
                    "epoch": epoch+1,
                    "train": {"loss": train_loss, "acc": train_acc, "f1": train_f1},
                    "val": {
                        "loss": val_loss, 
                        "acc": val_acc,
                        "F1": val_f1,
                        "AUC": val_auc
                    },
                })

                if config["multi"]:
                    log_json["runs"][protocol_key]["steps"][-1]["train"]["sub_loss"] = train_sub_loss
                    log_json["runs"][protocol_key]["steps"][-1]["train"]["sub_acc"] = train_sub_preds
                    log_json["runs"][protocol_key]["steps"][-1]["val"]["sub_loss"] = val_sub_loss
                    log_json["runs"][protocol_key]["steps"][-1]["val"]["sub_acc"] = val_sub_preds

                with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)

                # IMPROVEMENT: Check for F1 improvement too
                improved = False
                if val_acc > best_acc:
                    best_acc = val_acc
                    improved = True
                    
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    improved = True
                
                if improved:
                    best_results = f"Best Results (Epoch {epoch+1}) - Acc: {val_acc:.5f}, F1: {val_f1:.5f}, AUC: {val_auc:.5f}"
                    best_results_epoch = epoch+1
                    print(f"  ✓ New best model (epoch {epoch+1})!")
                    
                    # Reset early stopping counter
                    early_stop_counter = 0

                    # Generate classification report
                    report = classification_report(
                        val_labels.cpu().numpy(), 
                        val_preds.cpu().numpy(),
                        target_names=["truth", "deception"],
                        zero_division=0
                    )

                    # Log best metrics to wandb
                    wandb.run.summary["best_val_accuracy"] = val_acc
                    wandb.run.summary["best_val_f1"] = val_f1
                    wandb.run.summary["best_val_auc"] = val_auc
                    wandb.run.summary["best_epoch"] = epoch+1
                    
                    # Save model
                    model_path = os.path.join(config["log_dir"], f"best_model_{train_file.split('.')[0]}_{test_file.split('.')[0]}.pt")
                    torch.save(model.state_dict(), model_path)
                    print(f"  Model saved to {model_path}")
                else:
                    # IMPROVEMENT: Increment early stopping counter
                    early_stop_counter += 1
                    print(f"  Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                    
                    if early_stop_counter >= early_stop_patience:
                        print(f"  Early stopping triggered! No improvement for {early_stop_patience} epochs.")
                        break
            
            # Log final results
            print("\nTraining completed.")
            print(best_results)

            log_json["runs"][protocol_key]["best_results_epoch"] = best_results_epoch
            log_json["runs"][protocol_key]["report"] = report
            with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)

    except Exception as e:
        log_json["error"] = e.__str__()
        tb = traceback.format_exc()
        log_json["traceback"] = str(tb)
        with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)
        
        # Log error to wandb
        wandb.log({"error": e.__str__(), "traceback": str(tb)})
        
        raise e
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    train_and_evaluate()