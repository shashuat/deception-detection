import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.audio_visual_dataset import AudioVisualDataset, af_collate_fn
from models.audio_model import W2V2_Model
from models.fusion_model import Fusion
from models.visual_model import ViT_model
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc


# Configuration dictionary
config = {
    # Paths
    "data_root": "/Data/dec/DOLOS/",
    "audio_path": "/Data/dec/data/audio_files/",
    "visual_path": "/Data/dec/data/face_frames/",
    "log_dir": "logs",
    
    # Training parameters
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "lr": 1e-3,
    "batch_size": 16,
    "num_epochs": 20,
    
    # Model configuration
    "model_to_train": "fusion",  # Options: "audio", "vision", "fusion"
    "num_encoders": 4,
    "adapter": True,
    "adapter_type": "efficient_conv",  # Options: "nlp", "efficient_conv"
    "fusion_type": "cross2",  # Options: "concat", "cross2"
    "multi": False,  # Use multitask learning
    
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
    ]
}


def train_one_epoch(config, train_loader, model, optimizer, criterion, loss_audio=None, loss_vision=None):
    """Train model for one epoch"""
    model.train()
    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    start_time = time.time()
    
    if config["model_to_train"] == "audio":
        for i, (waves, _, labels) in enumerate(train_loader):
            # Prepare input
            waves = waves.squeeze(1).to(config["device"])
            labels = labels.to(config["device"])
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(waves)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track progress
            epoch_loss.append(loss.item())
            epoch_preds.append(torch.argmax(outputs, dim=1))
            epoch_labels.append(labels)
            
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item():.5f}")
                
    elif config["model_to_train"] == "vision":
        for i, (_, faces, labels) in enumerate(train_loader):
            # Prepare input
            faces = faces.to(config["device"])
            labels = labels.to(config["device"])
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(faces)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track progress
            epoch_loss.append(loss.item())
            epoch_preds.append(torch.argmax(outputs, dim=1))
            epoch_labels.append(labels)
            
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item():.5f}")
                
    else:  # Fusion model
        for i, (waves, faces, labels) in enumerate(train_loader):
            # Prepare input
            waves = waves.squeeze(1).to(config["device"])
            faces = faces.to(config["device"])
            labels = labels.to(config["device"])
            
            # Forward pass
            optimizer.zero_grad()
            outputs, a_outputs, v_outputs = model(waves, faces)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            if config["multi"] and a_outputs is not None and v_outputs is not None:
                a_loss = loss_audio(a_outputs, labels)
                v_loss = loss_vision(v_outputs, labels)
                loss = loss + a_loss + v_loss
            
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
    
    time_taken = time.time() - start_time
    print(f"Training completed in {time_taken:.2f} seconds")
    
    return avg_loss, epoch_preds, epoch_labels


def validate(config, val_loader, model, criterion, loss_audio=None, loss_vision=None):
    """Validate model on validation set"""
    model.eval()
    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    
    with torch.no_grad():
        if config["model_to_train"] == "audio":
            for waves, _, labels in val_loader:
                waves = waves.squeeze(1).to(config["device"])
                labels = labels.to(config["device"])
                
                outputs = model(waves)
                loss = criterion(outputs, labels)
                
                epoch_loss.append(loss.item())
                epoch_preds.append(torch.argmax(outputs, dim=1))
                epoch_labels.append(labels)
                
        elif config["model_to_train"] == "vision":
            for _, faces, labels in val_loader:
                faces = faces.to(config["device"])
                labels = labels.to(config["device"])
                
                outputs = model(faces)
                loss = criterion(outputs, labels)
                
                epoch_loss.append(loss.item())
                epoch_preds.append(torch.argmax(outputs, dim=1))
                epoch_labels.append(labels)
                
        else:  # Fusion model
            for waves, faces, labels in val_loader:
                waves = waves.squeeze(1).to(config["device"])
                faces = faces.to(config["device"])
                labels = labels.to(config["device"])
                
                outputs, a_outputs, v_outputs = model(waves, faces)
                
                loss = criterion(outputs, labels)
                if config["multi"] and a_outputs is not None and v_outputs is not None:
                    a_loss = loss_audio(a_outputs, labels)
                    v_loss = loss_vision(v_outputs, labels)
                    loss = loss + a_loss + v_loss
                
                epoch_loss.append(loss.item())
                epoch_preds.append(torch.argmax(outputs, dim=1))
                epoch_labels.append(labels)
    
    # Combine results
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)
    avg_loss = np.mean(epoch_loss)
    
    return avg_loss, epoch_preds, epoch_labels


def evaluate_metrics(labels, preds):
    """Calculate evaluation metrics"""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score


def train_and_evaluate():
    """Main training and evaluation function"""
    # Create log directory
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Create model name
    model_name = f"DOLOS_{config['model_to_train']}_Encoders_{config['num_encoders']}_Adapter_{config['adapter']}"
    if config["adapter"]:
        model_name += f"_type_{config['adapter_type']}"
    if config["model_to_train"] == "fusion":
        model_name += f"_fusion_{config['fusion_type']}"
    if config["multi"]:
        model_name += "_multi"
        
    # Create log file
    log_file = os.path.join(config["log_dir"], f"{int(time.time())}_{model_name}.txt")
    
    # Log configuration
    with open(log_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Learning Rate: {config['lr']}\n")
        f.write(f"Batch Size: {config['batch_size']}\n")
        f.write(f"Epochs: {config['num_epochs']}\n")
        f.write(f"Num Encoders: {config['num_encoders']}\n")
        f.write(f"Adapter: {config['adapter']}\n")
        if config["adapter"]:
            f.write(f"Adapter Type: {config['adapter_type']}\n")
        f.write("----------------------------------------\n\n")
    
    # Train and evaluate on each protocol
    for protocol in config["protocols"]:
        train_file, test_file = protocol
        
        print(f"\n\nRunning protocol: {train_file} (train), {test_file} (test)")
        
        # Update log file
        with open(log_file, 'a') as f:
            f.write(f"\nProtocol: {train_file} (train), {test_file} (test)\n")
        
        # Get annotation file paths
        train_anno = os.path.join(config["data_root"], "protocols", train_file)
        test_anno = os.path.join(config["data_root"], "protocols", test_file)
        
        # Create datasets
        # train_dataset = AudioVisualDataset(train_anno, config["audio_path"], config["visual_path"])
        # test_dataset = AudioVisualDataset(test_anno, config["audio_path"], config["visual_path"])
        
        train_dataset = AudioVisualDataset(
        train_anno, 
        config["audio_path"], 
        config["visual_path"], 
        ignore_audio_errors=True  # Set to True to continue even with audio errors
        )

        test_dataset = AudioVisualDataset(
            test_anno, 
            config["audio_path"], 
            config["visual_path"], 
            ignore_audio_errors=True  # Set to True to continue even with audio errors
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
        
        # Create model
        if config["model_to_train"] == "audio":
            model = W2V2_Model(config["num_encoders"], config["adapter"], config["adapter_type"])
        elif config["model_to_train"] == "vision":
            model = ViT_model(config["num_encoders"], config["adapter"], config["adapter_type"])
        else:
            model = Fusion(
                config["fusion_type"], 
                config["num_encoders"], 
                config["adapter"], 
                config["adapter_type"], 
                config["multi"]
            )
        
        # Move model to device
        model.to(config["device"])
        print(f"Model created: {model_name}")
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()
        
        # Additional loss functions for multitask learning
        loss_audio = loss_vision = None
        if config["multi"]:
            loss_audio = nn.CrossEntropyLoss()
            loss_vision = nn.CrossEntropyLoss()
            print("Multitask learning enabled")
        
        # Training loop
        best_acc = 0.0
        print(f"Starting training for {config['num_epochs']} epochs")
        
        for epoch in range(config["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            
            # Train for one epoch
            train_loss, train_preds, train_labels = train_one_epoch(
                config, train_loader, model, optimizer, criterion, loss_audio, loss_vision
            )
            
            # Calculate training metrics
            train_acc, train_f1, train_auc = evaluate_metrics(
                train_labels.cpu().numpy(), train_preds.cpu().numpy()
            )
            
            # Validate
            val_loss, val_preds, val_labels = validate(
                config, test_loader, model, criterion, loss_audio, loss_vision
            )
            
            # Calculate validation metrics
            val_acc, val_f1, val_auc = evaluate_metrics(
                val_labels.cpu().numpy(), val_preds.cpu().numpy()
            )
            
            # Print results
            print(f"Epoch {epoch+1} Results:")
            print(f"  Train - Loss: {train_loss:.5f}, Acc: {train_acc:.5f}, F1: {train_f1:.5f}, AUC: {train_auc:.5f}")
            print(f"  Valid - Loss: {val_loss:.5f}, Acc: {val_acc:.5f}, F1: {val_f1:.5f}, AUC: {val_auc:.5f}")
            
            # Update log file
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, ")
                f.write(f"Valid Loss: {val_loss:.5f}, Valid Acc: {val_acc:.5f}, Valid F1: {val_f1:.5f}, Valid AUC: {val_auc:.5f}\n")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_results = f"Best Results (Epoch {epoch+1}) - Acc: {val_acc:.5f}, F1: {val_f1:.5f}, AUC: {val_auc:.5f}"
                
                # Generate classification report
                report = classification_report(
                    val_labels.cpu().numpy(), 
                    val_preds.cpu().numpy(),
                    target_names=["truth", "deception"]
                )
                
                # Save model
                model_path = os.path.join(config["log_dir"], f"best_model_{train_file.split('.')[0]}_{test_file.split('.')[0]}.pt")
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved to {model_path}")
        
        # Log final results
        print("\nTraining completed.")
        print(best_results)
        
        with open(log_file, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(best_results + "\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n\n")


if __name__ == "__main__":
    train_and_evaluate()