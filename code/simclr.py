# simple_simclr.py
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import numpy as np

# Import your model components
from models.encoders.audio_encoder import WavEncoder
from models.encoders.visual_encoder import FaceEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.face_encoder import CombinedFace3DViT
from models.encoders.audio_encoder import WhisperTokenEncoder

# Import your dataloader
from dataloader.audio_visual_dataset import DatasetDOLOS

# Get environment paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from setup import get_env

ENV = get_env()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------------
# Configuration
#--------------------------------------------------------
config = {
    "data_root": ENV["DOLOS_PATH"],
    "audio_path": ENV["AUDIO_PATH"],
    "visual_path": ENV["VISUAL_PATH"],
    "transcripts_path": ENV["TRANSCRIPTS_PATH"],
    "output_dir": os.path.join(ENV["LOGS_PATH"], "simclr_pretraining"),
    "batch_size": 16,
    "num_epochs": 50,
    "lr": 1e-4,
    "temperature": 0.1,
    "num_layers": 4,
    "adapter": True,
    "adapter_type": "efficient_conv",
    "modalities": ["audio", "faces", "text", "whisper"],
    "pretrain_protocol": ["train_fold1.csv"]  # Use just one fold for simplicity
}

os.makedirs(config["output_dir"], exist_ok=True)

class SimpleContrastiveDataset(Dataset):
    """
    A simple dataset wrapper that creates two augmented views of each sample
    using random transformations.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def simple_augment(self, tensor):
        """Basic augmentation by adding noise"""
        if tensor is None:
            return None
        
        # Make a copy to avoid modifying the original
        augmented = tensor.clone()
        
        # Add random noise
        noise = torch.randn_like(augmented) * 0.1
        augmented = augmented + noise
        
        return augmented
    
    def __getitem__(self, idx):
        # Get original sample
        sample = self.base_dataset[idx]
        
        # Handle different dataset return formats
        if isinstance(sample, tuple):
            if len(sample) >= 2:
                data, label = sample[0], sample[1]
            else:
                raise ValueError(f"Unexpected dataset return format: {sample}")
        else:
            # If it's not a tuple, assume it's the data itself
            data = sample
            label = torch.zeros(1)  # Dummy label
        
        # Create two augmented views
        if isinstance(data, dict):
            # Handle dictionary format (with modality keys)
            view1 = {k: self.simple_augment(v) for k, v in data.items()}
            view2 = {k: self.simple_augment(v) for k, v in data.items()}
        else:
            # Handle tensor format directly
            view1 = self.simple_augment(data)
            view2 = self.simple_augment(data)
        
        return (view1, view2), label


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super(ProjectionMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # If input is 3D [B, S, D], take mean along sequence dimension
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)  # [B, D]
        
        x = self.layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class SimpleSimCLR(nn.Module):
    def __init__(self, config):
        super(SimpleSimCLR, self).__init__()
        self.modalities = config["modalities"]
        self.num_layers = config["num_layers"]
        self.adapter = config["adapter"]
        self.adapter_type = config["adapter_type"]
        
        # Initialize appropriate encoders
        self.encoders = nn.ModuleDict()
        
        if "audio" in self.modalities:
            self.encoders["audio"] = WavEncoder(self.num_layers, self.adapter, self.adapter_type)
        
        if "faces" in self.modalities:
            self.encoders["faces"] = FaceEncoder(self.num_layers, self.adapter, self.adapter_type)
        
        if "text" in self.modalities:
            self.encoders["text"] = TextEncoder(embedding_dim=768, transformer_layers=self.num_layers)
        
        if "whisper" in self.modalities:
            self.encoders["whisper"] = WhisperTokenEncoder(
                embedding_dim=768, 
                num_layers=self.num_layers, 
                num_heads=6, 
                hidden_dim=1024
            )
        
        # Projection MLPs for each modality
        self.projectors = nn.ModuleDict({
            mod: ProjectionMLP(input_dim=768) for mod in self.modalities
        })
    
    def forward(self, x):
        """
        Forward pass through encoders and projectors.
        
        Args:
            x: Either a dictionary with modality keys or a single tensor if using only one modality
        
        Returns:
            Dictionary of projected embeddings for each modality
        """
        embeddings = {}
        projections = {}
        
        if isinstance(x, dict):
            # Process each modality if input is a dictionary
            for mod in self.modalities:
                if mod in x and x[mod] is not None and mod in self.encoders:
                    # Process through encoder
                    embeddings[mod] = self.encoders[mod](x[mod])
                    # Project and normalize
                    proj = self.projectors[mod](embeddings[mod])
                    projections[mod] = F.normalize(proj, dim=1)
        else:
            # If input is a single tensor, use the first encoder
            # (Assuming we're only working with one modality)
            mod = self.modalities[0]
            embeddings[mod] = self.encoders[mod](x)
            proj = self.projectors[mod](embeddings[mod])
            projections[mod] = F.normalize(proj, dim=1)
        
        return projections


def nt_xent_loss(z_i, z_j, temperature=0.1):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
    """
    batch_size = z_i.shape[0]
    
    # Cosine similarity between all combinations of samples
    z = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    
    # Mask for positive pairs
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    
    # We use these positive pairs as targets
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0)
    
    # Mask out self-similarity
    mask = torch.ones_like(sim)
    mask = mask.fill_diagonal_(0)
    mask[:batch_size, :batch_size] = 0
    mask[batch_size:, batch_size:] = 0
    
    # Remove self-similarity from denominator
    sim = mask * sim
    
    # InfoNCE loss
    negatives = torch.exp(sim)
    row_sum = negatives.sum(dim=1)
    log_prob = positive_samples - torch.log(row_sum)
    
    loss = -log_prob.mean()
    return loss


def train_simclr(config):
    print(f"Using device: {DEVICE}")
    
    # Create dataset for pretraining
    train_datasets = []
    for protocol_file in config["pretrain_protocol"]:
        anno_path = os.path.join(config["data_root"], "protocols", protocol_file)
        
        if not os.path.exists(anno_path):
            print(f"Warning: Protocol file {anno_path} not found.")
            continue
        
        dataset = DatasetDOLOS(
            annotations_file=anno_path,
            audio_dir=config["audio_path"],
            img_dir=config["visual_path"],
            transcripts_dir=config["transcripts_path"],
            ignore_audio_errors=True
        )
        
        train_datasets.append(dataset)
    
    if not train_datasets:
        raise ValueError("No valid protocol files found.")
    
    print(f"Loaded {len(train_datasets)} datasets")
    
    # If only one dataset, use it directly
    if len(train_datasets) == 1:
        base_dataset = train_datasets[0]
    else:
        # Combine datasets
        from torch.utils.data import ConcatDataset
        base_dataset = ConcatDataset(train_datasets)
    
    # Create contrastive dataset
    contrastive_dataset = SimpleContrastiveDataset(base_dataset)
    print(f"Total training samples: {len(contrastive_dataset)}")
    
    # Simple custom collate function
    def custom_collate(batch):
        views_pairs, labels = zip(*batch)
        views1, views2 = zip(*views_pairs)
        
        # Handle dictionary data format
        if isinstance(views1[0], dict):
            # Collect all modalities
            collated_view1 = {}
            collated_view2 = {}
            
            # Get all keys
            all_keys = set()
            for v in views1:
                if isinstance(v, dict):
                    all_keys.update(v.keys())
            
            # Collect and stack tensors for each modality
            for key in all_keys:
                view1_tensors = [v[key] for v in views1 if isinstance(v, dict) and key in v]
                view2_tensors = [v[key] for v in views2 if isinstance(v, dict) and key in v]
                
                if view1_tensors and view2_tensors:
                    try:
                        collated_view1[key] = torch.stack(view1_tensors)
                        collated_view2[key] = torch.stack(view2_tensors)
                    except:
                        print(f"Error stacking tensors for {key}")
            
            views1 = collated_view1
            views2 = collated_view2
        else:
            # Handle tensor data format (single modality)
            try:
                views1 = torch.stack(views1)
                views2 = torch.stack(views2)
            except:
                print("Error stacking tensor views")
                # Return empty batch as fallback
                return (({}, {}), torch.tensor([]))
        
        # Convert labels to tensor
        labels = torch.tensor(labels)
        
        return ((views1, views2), labels)
    
    # Create data loader
    train_loader = DataLoader(
        contrastive_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate,
        drop_last=True
    )
    
    # Create model
    model = SimpleSimCLR(config).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    # Simple step learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, ((view1, view2), _) in enumerate(progress_bar):
            # Skip empty batches
            if not view1 or not view2:
                continue
            
            # Move data to device
            if isinstance(view1, dict):
                view1 = {k: v.to(DEVICE) for k, v in view1.items()}
                view2 = {k: v.to(DEVICE) for k, v in view2.items()}
            else:
                view1 = view1.to(DEVICE)
                view2 = view2.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            proj1 = model(view1)
            proj2 = model(view2)
            
            # Calculate loss for each modality
            loss = 0
            count = 0
            
            # Process each modality that is present in both views
            for mod in proj1.keys():
                if mod in proj2:
                    mod_loss = nt_xent_loss(proj1[mod], proj2[mod], config["temperature"])
                    loss += mod_loss
                    count += 1
            
            # If no matching modalities, skip batch
            if count == 0:
                continue
            
            # Average loss across modalities
            loss = loss / count
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update stats
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.5f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Save encoder weights for easy loading
            encoder_state = {}
            for mod, encoder in model.encoders.items():
                encoder_state[mod] = encoder.state_dict()
            
            encoder_path = os.path.join(config["output_dir"], "simclr_encoders_best.pt")
            torch.save(encoder_state, encoder_path)
            
            print(f"New best model saved (loss: {best_loss:.5f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config["output_dir"], f"simclr_checkpoint_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
    
    # Save final encoders
    final_encoder_state = {}
    for mod, encoder in model.encoders.items():
        final_encoder_state[mod] = encoder.state_dict()
    
    final_encoder_path = os.path.join(config["output_dir"], "simclr_encoders_final.pt")
    torch.save(final_encoder_state, final_encoder_path)
    
    print(f"Final encoders saved at: {final_encoder_path}")
    return final_encoder_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple SimCLR Pretraining")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size")
    parser.add_argument("--epochs", type=int, default=config["num_epochs"], help="Number of epochs")
    parser.add_argument("--modality", type=str, default=None, help="Use only one specific modality")
    
    args = parser.parse_args()
    
    # Update config with args
    config["batch_size"] = args.batch_size
    config["num_epochs"] = args.epochs
    
    if args.modality:
        # Use only the specified modality
        config["modalities"] = [args.modality]
        print(f"Using only the {args.modality} modality")
    
    try:
        encoder_path = train_simclr(config)
        print(f"Pretraining complete! Encoder weights saved at: {encoder_path}")
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()