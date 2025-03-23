# train_test.py with improvements

import os
import sys
import time
import json
import traceback
import torch
import numpy as np
import wandb

# Import standard modules
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc

# Import dataset
from dataloader.audio_visual_dataset import DatasetDOLOS, af_collate_fn

# Import models
from models.improved_fusion_model import ImprovedFusion  # Use the improved fusion model

# Import new training utilities
from training.improved_training import train_with_improvements, validate_with_improvements
from utils.focal_loss import FocalLoss, LabelSmoothingLoss

# Path setup
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
    
    # Training parameters
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr": 5e-4,  # Slightly lower learning rate
    "batch_size": 8,
    "effective_batch_size": 32,  # Using gradient accumulation to achieve this
    "accumulation_steps": 4,  # batch_size * accumulation_steps = effective_batch_size
    "num_epochs": 30,  
    "warmup_epochs": 2,  
    "weight_decay": 2e-5,  
    
    # Model configuration
    "modalities": ["faces", "audio", "text", "whisper"],
    "num_layers": 4,
    "adapter": True,
    "adapter_type": "efficient_conv",
    "fusion_type": "cross_attention",
    "multi": True,
    "sub_labels": False,
    
    # New training features
    "use_focal_loss": True,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "use_label_smoothing": True,
    "label_smoothing": 0.1,
    "use_mixup": True,
    "mixup_alpha": 0.2,
    "use_amp": True,  # Automatic mixed precision
    
    # Protocols for training and testing
    "protocols": [
        ["train_fold3.csv", "test_fold3.csv"],
        # Add more protocols as needed
    ],
    
    # wandb configuration
    "wandb": {
        "project": "multimodal-lie-detection",
        "entity": None,
        "tags": ["multimodal", "lie-detection", "deep-learning", "improved-fusion"]
    }
}

def train_and_evaluate():
    """Main training and evaluation function with improvements"""
    # Initialize wandb with enhanced configuration
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
    if config["use_focal_loss"]:
        model_name += "_focal"
    if config["use_label_smoothing"]:
        model_name += "_smooth"
    if config["use_mixup"]:
        model_name += "_mixup"
    
    # Set wandb run name
    wandb.run.name = model_name
        
    # Create log file
    log_file = os.path.join(config["log_dir"], f"{int(time.time())}_{model_name}.json")
    log_json = {
        "modalities": config["modalities"],
        "learning_rate": config["lr"],
        "batch_size": config["batch_size"],
        "effective_batch_size": config["effective_batch_size"],
        "num_epochs": config["num_epochs"],
        "num_layers": config["num_layers"],
        "adapter": config["adapter"],
        "focal_loss": config["use_focal_loss"],
        "label_smoothing": config["use_label_smoothing"],
        "mixup": config["use_mixup"],
        "amp": config["use_amp"],
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
                raise Exception(f"Protocol {protocol_key} was already run")
            
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
            
            # Create improved fusion model
            model = ImprovedFusion(
                config["fusion_type"],
                config["modalities"],
                config["num_layers"], 
                config["adapter"], 
                config["adapter_type"], 
                config["multi"],
                train_dataset.num_sub_labels,
                dropout=0.25  # Higher dropout for improved regularization
            )
            
            # Move model to device
            model.to_device(config["device"])
            print(f"Model created: {model_name}")
            
            # Create optimizer with weight decay
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                betas=(0.9, 0.999)  # Typical AdamW betas
            )
            
            # Improved learning rate scheduler
            from torch.optim.lr_scheduler import OneCycleLR
            
            # Calculate total steps for lr scheduling
            total_steps = config["num_epochs"] * len(train_loader) // config["accumulation_steps"]
            
            # Use OneCycleLR for better convergence
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config["lr"],
                total_steps=total_steps,
                pct_start=0.2,  # Percentage of steps for warmup
                div_factor=25,  # Initial lr = max_lr/div_factor
                final_div_factor=1000,  # Final lr = max_lr/final_div_factor
                anneal_strategy='cos'
            )
            
            # Create loss functions based on configuration
            if config["use_focal_loss"]:
                criterion = FocalLoss(
                    alpha=config["focal_alpha"],
                    gamma=config["focal_gamma"],
                    device=config["device"]
                )
                print("Using Focal Loss")
            elif config["use_label_smoothing"]:
                criterion = LabelSmoothingLoss(smoothing=config["label_smoothing"])
                print("Using Label Smoothing Loss")
            else:
                criterion = torch.nn.CrossEntropyLoss()
                print("Using Cross Entropy Loss")
            
            # Additional loss functions for multitask learning
            secondary_loss = {}
            if config["multi"]:
                for k in config["modalities"]:
                    if config["use_focal_loss"]:
                        secondary_loss[k] = FocalLoss(
                            alpha=config["focal_alpha"],
                            gamma=config["focal_gamma"],
                            device=config["device"]
                        )
                    elif config["use_label_smoothing"]:
                        secondary_loss[k] = LabelSmoothingLoss(smoothing=config["label_smoothing"])
                    else:
                        secondary_loss[k] = torch.nn.CrossEntropyLoss()
                print("Multitask learning enabled")

            # Loss functions for sub-labels
            sub_labels_loss = []
            if config["sub_labels"]:
                for _ in range(train_dataset.num_sub_labels):
                    if config["use_focal_loss"]:
                        sub_labels_loss.append(FocalLoss(
                            alpha=config["focal_alpha"],
                            gamma=config["focal_gamma"],
                            device=config["device"]
                        ))
                    elif config["use_label_smoothing"]:
                        sub_labels_loss.append(LabelSmoothingLoss(smoothing=config["label_smoothing"]))
                    else:
                        sub_labels_loss.append(torch.nn.CrossEntropyLoss())
                print(f"Sub-labels enabled: {train_dataset.num_sub_labels}")
            
            # Early stopping parameters
            early_stop_patience = 10
            early_stop_counter = 0
            best_acc = 0.0
            best_f1 = 0.0
            best_results = ""
            best_results_epoch = -1
            
            # Create directories for saving models
            model_dir = os.path.join(config["log_dir"], "models")
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"Starting training for {config['num_epochs']} epochs")
            
            for epoch in range(config["num_epochs"]):
                print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
                
                # Train with improved function
                train_loss, train_preds, train_labels, train_sub_loss, train_sub_preds, sub_labels_loss_values = train_with_improvements(
                    config, train_loader, model, optimizer, criterion, secondary_loss, sub_labels_loss, 
                    scheduler=scheduler,
                    use_mixup=config["use_mixup"],
                    mixup_alpha=config["mixup_alpha"],
                    use_amp=config["use_amp"],
                    accumulation_steps=config["accumulation_steps"]
                )
                
                # Calculate training metrics
                train_acc, train_f1, train_auc = evaluate_metrics(
                    train_labels.cpu().numpy(), train_preds.cpu().numpy()
                )

                # Multi-task training metrics
                if config["multi"]:
                    for k in train_sub_preds:
                        train_sub_preds[k] = evaluate_metrics(
                            train_labels.cpu().numpy(), train_sub_preds[k].cpu().numpy()
                        )
                
                # Validate with improved function
                val_loss, val_preds, val_labels, val_sub_loss, val_sub_preds = validate_with_improvements(
                    config, test_loader, model, criterion, secondary_loss, 
                    use_amp=config["use_amp"]
                )
                
                # Calculate validation metrics
                val_acc, val_f1, val_auc = evaluate_metrics(
                    val_labels.cpu().numpy(), val_preds.cpu().numpy()
                )

                # Multi-task validation metrics
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
                print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Update log file
                log_json["runs"][protocol_key]["steps"].append({
                    "epoch": epoch+1,
                    "train": {"loss": train_loss, "acc": train_acc, "f1": train_f1, "auc": train_auc},
                    "val": {"loss": val_loss, "acc": val_acc, "f1": val_f1, "auc": val_auc},
                })

                if config["multi"]:
                    log_json["runs"][protocol_key]["steps"][-1]["train"]["sub_loss"] = train_sub_loss
                    log_json["runs"][protocol_key]["steps"][-1]["train"]["sub_acc"] = train_sub_preds
                    log_json["runs"][protocol_key]["steps"][-1]["val"]["sub_loss"] = val_sub_loss
                    log_json["runs"][protocol_key]["steps"][-1]["val"]["sub_acc"] = val_sub_preds

                with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)

                # Save based on better F1 score (more robust than accuracy)
                improvement = False
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    improvement = True
                    
                if val_acc > best_acc:
                    best_acc = val_acc
                    if not improvement:  # Only count as improvement if F1 didn't improve
                        improvement = True
                
                if improvement:
                    best_results = f"Best Results (Epoch {epoch+1}) - Acc: {val_acc:.5f}, F1: {val_f1:.5f}, AUC: {val_auc:.5f}"
                    best_results_epoch = epoch+1
                    print(f"  ✓ New best model (epoch {epoch+1})!")
                    
                    # Reset early stopping counter
                    early_stop_counter = 0
                    
                    # Create classification report for detailed metrics
                    report = classification_report(
                        val_labels.cpu().numpy(), 
                        val_preds.cpu().numpy(),
                        target_names=["truth", "deception"],
                        output_dict=True,
                        zero_division=0
                    )
                    
                    # Log best metrics to wandb
                    wandb.run.summary["best_val_accuracy"] = val_acc
                    wandb.run.summary["best_val_f1"] = val_f1
                    wandb.run.summary["best_val_auc"] = val_auc
                    wandb.run.summary["best_epoch"] = epoch+1
                    
                    # Log confusion matrix
                    try:
                        wandb.log({
                            "confusion_matrix": wandb.plot.confusion_matrix(
                                y_true=val_labels.cpu().numpy(), 
                                preds=val_preds.cpu().numpy(),
                                class_names=["truth", "deception"]
                            )
                        })
                    except Exception as cm_error:
                        print(f"Error creating confusion matrix: {cm_error}")
                    
                    # Save model
                    model_path = os.path.join(
                        model_dir, 
                        f"best_model_{train_file.split('.')[0]}_{test_file.split('.')[0]}.pt"
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'best_f1': best_f1,
                        'config': config
                    }, model_path)
                    print(f"  Model saved to {model_path}")
                    
                    # Add to log
                    log_json["runs"][protocol_key]["best_model_path"] = model_path
                    log_json["runs"][protocol_key]["best_results"] = {
                        "epoch": epoch+1,
                        "accuracy": val_acc,
                        "f1": val_f1,
                        "auc": val_auc,
                        "report": report
                    }
                    with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)
                else:
                    # Increment early stopping counter
                    early_stop_counter += 1
                    print(f"  Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                    
                    if early_stop_counter >= early_stop_patience:
                        print(f"  Early stopping triggered! No improvement for {early_stop_patience} epochs.")
                        break
            
            # Log final results after protocol completion
            print("\nTraining completed for this protocol.")
            print(best_results)
            
            # Prepare for next protocol
            # Free up memory
            del model
            del optimizer
            del scheduler
            torch.cuda.empty_cache()

    except Exception as e:
        log_json["error"] = str(e)
        tb = traceback.format_exc()
        log_json["traceback"] = tb
        with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)
        
        # Log error to wandb
        wandb.log({"error": str(e), "traceback": tb})
        
        raise e
    
    finally:
        wandb.finish()


def evaluate_metrics(labels, preds):
    """Calculate evaluation metrics unchanged"""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score


if __name__ == "__main__":
    train_and_evaluate()