# train_test.py with wandb integration
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
import wandb  # Import wandb

from dataloader.audio_visual_dataset import DatasetDOLOS, af_collate_fn
from models.audio_model import W2V2_Model
from models.fusion_model import Fusion
from models.visual_model import ViT_model

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
    "lr": 1e-4,
    "batch_size": 8,
    "num_epochs": 20,
    "warmup_epochs": 1,  # warmup epochs for scheduler

    # Model configuration
    "modalities": ["faces", "audio", "text", "whisper"],
    "num_layers": 4,
    "adapter": True,
    "adapter_type": "efficient_conv",  # Options: "nlp", "efficient_conv"
    "fusion_type": "cross_attention",  # Options: "concat", "cross2", "cross_attention"
    "multi": True,  # Use multitask learning
    "sub_labels": False, # Use sub labels (smile, cry, etc.)
    
    # Protocols for training and testing
    "protocols": [
        ["train_fold3.csv", "test_fold3.csv"],
        # Uncomment to add more protocols
        # ["train_fold1.csv", "test_fold1.csv"],
        # ["train_fold2.csv", "test_fold2.csv"],
        # ["long.csv", "short.csv"],
        # ["short.csv", "long.csv"],
        # ["male.csv", "female.csv"],
        # ["female.csv", "male.csv"],
    ],
    
    # wandb configuration
    "wandb": {
        "project": "multimodal-lie-detection",
        "entity": None,  # Change to your wandb username or team name
        "tags": ["multimodal", "lie-detection", "deep-learning"]
    }
}

def train_one_epoch(config, train_loader, model, optimizer, criterion, sub_loss, sub_labels_loss):
    """Train model for one epoch"""
    model.train()
    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    epoch_subloss = []
    epoch_subpreds = {}
    epoch_sub_labels_loss = []
    start_time = time.time()
    
    for i, (train_data, labels, sub_labels) in enumerate(train_loader):
        # Prepare input
        # waves = waves.squeeze(1).to(config["device"]) 
        # faces = faces
        # whisper_tokens = whisper_tokens.to(config["device"])
        # bert_embedding = bert_embedding.to(config["device"])

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
        optimizer.step()
        
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

    epoch_sub_labels_loss_tensor = torch.tensor(epoch_sub_labels_loss) if epoch_sub_labels_loss else torch.tensor([])  # Shape: [num_batches, num_heads]
    mean_loss_per_head = epoch_sub_labels_loss_tensor.mean(dim=0) if epoch_sub_labels_loss_tensor.numel() > 0 else torch.tensor([])  # Shape: [num_heads]
    mean_loss_per_head_list = mean_loss_per_head.tolist() if mean_loss_per_head.numel() > 0 else []
    
    return avg_loss, epoch_preds, epoch_labels, avg_sub_loss, epoch_subpreds, mean_loss_per_head_list


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
            # waves = waves.squeeze(1).to(config["device"])
            # faces = faces.to(config["device"])
            # labels = labels.to(config["device"])
            # whisper_tokens = whisper_tokens.to(config["device"])
            # bert_embedding = bert_embedding.to(config["device"])

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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer and 0, with a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_and_evaluate():
    """Main training and evaluation function"""
    # Initialize wandb with configuration
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        tags=config["wandb"]["tags"],
        config={k: v for k, v in config.items() if k != "wandb"},  # Exclude wandb config itself
        settings=wandb.Settings(start_method="thread")  # Use thread mode for better compatibility
    )
    
    # Create log directory
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Create model name
    model_name = f"DOLOS_layers_{config['num_layers']}_Adapter_{config['adapter']}"
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
        "num_pochs": config["num_epochs"],
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
            # Set protocol in wandb config instead of logging as a metric
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
            # Log dataset sizes, just to be sure
            wandb.log({
                "dataset/train_samples": len(train_dataset),
                "dataset/test_samples": len(test_dataset)
            })
            
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
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            
            # Improved LR scheduler: Cosine annealing with warmup
            total_steps = config["num_epochs"] * len(train_loader)
            warmup_steps = config["warmup_epochs"] * len(train_loader)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr=1e-6  # Minimum learning rate at the end of schedule
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
            print(f"Starting training for {config['num_epochs']} epochs")
            
            # Create a local directory for saving models
            local_model_dir = os.path.join(os.getcwd(), "wandb_models")
            os.makedirs(local_model_dir, exist_ok=True)
            
            for epoch in range(config["num_epochs"]):
                print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
                
                # Train for one epoch
                train_loss, train_preds, train_labels, train_sub_loss, train_sub_preds, sub_labels_loss_values = train_one_epoch(
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
                
                # Update scheduler at each step
                scheduler.step()
                
                # Print results
                print(f"Epoch {epoch+1} Results:")
                print(f"  Train - Loss: {train_loss:.5f}, Acc: {train_acc:.5f}, F1: {train_f1:.5f}, AUC: {train_auc:.5f}")
                print(f"  Valid - Loss: {val_loss:.5f}, Acc: {val_acc:.5f}, F1: {val_f1:.5f}, AUC: {val_auc:.5f}")
                print(f"  Train - Sub-Loss: {train_sub_loss} | {train_sub_preds}")
                print(f"  Valid - Sub-Loss: {val_sub_loss} | {val_sub_preds}")
                print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")

                if sub_labels_loss_values:
                    print(f"Train - sub labels loss: {sub_labels_loss_values}")
                
                # Update log file
                log_json["runs"][protocol_key]["steps"].append({
                    "epoch": epoch+1,
                    "train": {"loss": train_loss, "acc": train_acc},
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

                # Save best model - using a local path for wandb compatibility
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_results = f"Best Results (Epoch {epoch+1}) - Acc: {val_acc:.5f}, F1: {val_f1:.5f}, AUC: {val_auc:.5f}"
                    best_results_epoch = epoch+1
                    
                    # Generate classification report
                    report = classification_report(
                        val_labels.cpu().numpy(), 
                        val_preds.cpu().numpy(),
                        target_names=["truth", "deception"],
                        zero_division=0  # Handle the UndefinedMetricWarning
                    )
                    
                    # Log best metrics to wandb as summary values
                    wandb.run.summary["best_val_accuracy"] = val_acc
                    wandb.run.summary["best_val_f1"] = val_f1
                    wandb.run.summary["best_val_auc"] = val_auc
                    wandb.run.summary["best_epoch"] = epoch+1
                    
                    # Log confusion matrix to wandb - using the wandb builtin function
                    try:
                        wandb.log({
                            "confusion_matrix": wandb.plot.confusion_matrix(
                                y_true=val_labels.cpu().numpy(), 
                                preds=val_preds.cpu().numpy()
                            )
                        })
                    except Exception as cm_error:
                        print(f"Error creating confusion matrix: {cm_error}")
                        
                        # Fallback to simple metrics if the confusion matrix fails
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(val_labels.cpu().numpy(), val_preds.cpu().numpy())
                        wandb.log({
                            "confusion_matrix/true_negative": cm[0][0],
                            "confusion_matrix/false_positive": cm[0][1],
                            "confusion_matrix/false_negative": cm[1][0],
                            "confusion_matrix/true_positive": cm[1][1]
                        })
                    
                    # Just log the AUC score
                    wandb.run.summary["best_auc_score"] = val_auc
                    
                    # Save model to local directory for wandb compatibility
                    local_model_path = os.path.join(local_model_dir, f"best_model_{train_file.split('.')[0]}_{test_file.split('.')[0]}.pt")
                    torch.save(model.state_dict(), local_model_path)
                    print(f"New best model saved to {local_model_path}")
                    
                    # Save a copy to the main log directory as well (but don't use with wandb.save)
                    original_model_path = os.path.join(config["log_dir"], f"best_model_{train_file.split('.')[0]}_{test_file.split('.')[0]}.pt")
                    torch.save(model.state_dict(), original_model_path)
                    
                    # Log the model to wandb - using local path
                    # wandb.save(local_model_path)  # Commented out - will be uncommented when best model is found
            
            # Log final results
            print("\nTraining completed.")
            print(best_results)

            # Create and log the classification report
            try:
                report_data = classification_report(
                    val_labels.cpu().numpy(),
                    val_preds.cpu().numpy(),
                    target_names=["truth", "deception"],
                    output_dict=True,
                    zero_division=0  # Handle the UndefinedMetricWarning
                )
                
                # Store the report as a text artifact within wandb directory
                report_path = os.path.join(os.getcwd(), f"classification_report_{int(time.time())}.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                
                # Log the report file
                # wandb.save(report_path)  # Commented out - will be uncommented when best model is found
                
                # Also log key metrics as summaries
                wandb.run.summary["final_truth_f1"] = report_data["truth"]["f1-score"]
                wandb.run.summary["final_deception_f1"] = report_data["deception"]["f1-score"]
                wandb.run.summary["final_macro_f1"] = report_data["macro avg"]["f1-score"]
            except Exception as report_err:
                print(f"Error logging classification report: {report_err}")

            log_json["runs"][protocol_key]["best_results_epoch"] = best_results_epoch
            log_json["runs"][protocol_key]["report"] = report

            with open(log_file, 'w') as f: json.dump(log_json, f, indent=4)

    except Exception as e: # before raise exception, save it in the log.json
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