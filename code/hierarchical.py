import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import json
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from setup import get_env

ENV = get_env()
config = {
    # Paths
    "data_root": ENV["DOLOS_PATH"],
    "embeddings_dir": os.path.join(ENV["LOGS_PATH"], "embeddings"),
    "hierarchical_models_dir": os.path.join(ENV["LOGS_PATH"], "hierarchical_models"),
    
    # Model configuration
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "sequence_length": 10,  # Number of chunks to include in each sequence
    "embedding_dim": 3072,  # Actual dimension of each embedding (Changed to match the embeddings)
    "num_modalities": 1,    # Simplified to treat the embeddings as a single modality
    
    # Hierarchical model configuration
    "transformer_layers": 4,
    "transformer_heads": 8,
    "transformer_dim_feedforward": 2048,
    "dropout": 0.2,
    
    # Protocols for training the hierarchical model
    "protocols": [
        ["train_fold3.csv", "test_fold3.csv"],
        # Add more protocols as needed
    ],
    
    # wandb configuration
    "wandb": {
        "project": "multimodal-lie-detection-hierarchical",
        "entity": None,
        "tags": ["hierarchical", "lie-detection", "sequence-learning"]
    }
}

class SequenceDataset(Dataset):
    """Dataset for sequence-level training with latent embeddings"""
    
    def __init__(self, embeddings_path, sequence_length=10, token_aggregation='mean', mode='random'):
        """
        Initialize sequence dataset from saved embeddings
        
        Args:
            embeddings_path: Path to the H5 file with embeddings
            sequence_length: Number of chunks in each sequence
            token_aggregation: How to aggregate tokens in each sample - 'mean', 'first', or 'all'
            mode: How to create sequences - 'random', 'consecutive', or 'grouped_by_video'
        """
        self.embeddings_path = embeddings_path
        self.sequence_length = sequence_length
        self.token_aggregation = token_aggregation
        self.mode = mode
        
        # Check if file exists
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        # Load embeddings
        with h5py.File(embeddings_path, 'r') as f:
            # Get the number of samples
            if 'embeddings' in f:
                self.num_samples = f.attrs.get('num_samples', len(f['embeddings']))
                # Check the shape of embeddings
                embedding_shape = f['embeddings'].shape
                print(f"Embeddings shape in {os.path.basename(embeddings_path)}: {embedding_shape}")
                self.num_tokens = embedding_shape[1]  # Number of tokens per sample (64)
                self.embedding_dim = embedding_shape[2]  # Dimension of each token (3072)
            else:
                raise ValueError(f"No embeddings found in {embeddings_path}")
            
            # Create sequence indices based on the mode
            if mode == 'random':
                self.sequences = self._create_random_sequences()
            elif mode == 'consecutive':
                self.sequences = self._create_consecutive_sequences()
            elif mode == 'grouped_by_video':
                # Extract video IDs if available
                if 'video_ids' in f:
                    video_ids = [f['video_ids'][i].decode('utf-8') for i in range(len(f['video_ids']))]
                    self.sequences = self._create_video_grouped_sequences(video_ids)
                else:
                    print("No video IDs found, falling back to random sequences")
                    self.sequences = self._create_random_sequences()
            else:
                raise ValueError(f"Unknown sequence creation mode: {mode}")
        
        print(f"Created {len(self.sequences)} sequences of length {sequence_length} from {self.num_samples} samples")
        print(f"Token aggregation method: {token_aggregation}")
                
    def _create_random_sequences(self):
        """Create random sequences of indices"""
        # Generate all possible indices
        all_indices = list(range(self.num_samples))
        
        # Create sequences by randomly sampling indices
        sequences = []
        num_sequences = self.num_samples // self.sequence_length
        
        for _ in range(num_sequences):
            # Sample sequence_length indices without replacement
            if len(all_indices) >= self.sequence_length:
                seq_indices = random.sample(all_indices, self.sequence_length)
                sequences.append(seq_indices)
            
        return sequences
    
    def _create_consecutive_sequences(self):
        """Create consecutive sequences of indices"""
        sequences = []
        # Create sequences of consecutive indices
        for i in range(0, self.num_samples - self.sequence_length + 1, self.sequence_length):
            sequences.append(list(range(i, i + self.sequence_length)))
        return sequences
    
    def _create_video_grouped_sequences(self, video_ids):
        """Create sequences grouped by video ID"""
        # Group indices by video ID
        video_to_indices = {}
        for i, vid in enumerate(video_ids):
            if vid not in video_to_indices:
                video_to_indices[vid] = []
            video_to_indices[vid].append(i)
        
        sequences = []
        # For each video, create sequences
        for vid, indices in video_to_indices.items():
            if len(indices) >= self.sequence_length:
                # Create sequences from consecutive frames in the same video
                for i in range(0, len(indices) - self.sequence_length + 1, self.sequence_length):
                    sequences.append(indices[i:i+self.sequence_length])
            else:
                # If not enough frames, pad with repeats
                if len(indices) > 0:  # Only if we have some frames
                    seq = indices.copy()
                    while len(seq) < self.sequence_length:
                        seq.append(random.choice(indices))
                    sequences.append(seq)
        
        return sequences
    
    def _aggregate_tokens(self, embeddings):
        """
        Aggregate tokens in each sample based on the specified method
        
        Args:
            embeddings: Array of shape [sequence_length, num_tokens, embedding_dim]
            
        Returns:
            Aggregated embeddings of shape [sequence_length, embedding_dim]
        """
        if self.token_aggregation == 'mean':
            # Average over all tokens (dimension 1)
            return np.mean(embeddings, axis=1)
        elif self.token_aggregation == 'first':
            # Take only the first token from each sample
            return embeddings[:, 0, :]
        elif self.token_aggregation == 'all':
            # Return all tokens, but reshaped to have a single sequence dimension
            # This changes the sequence length from [sequence_length] to [sequence_length * num_tokens]
            return embeddings.reshape(-1, self.embedding_dim)
        else:
            raise ValueError(f"Unknown token aggregation method: {self.token_aggregation}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        with h5py.File(self.embeddings_path, 'r') as f:
            sequence_indices = self.sequences[idx]
            
            # Extract embeddings for sequence - shape: [sequence_length, num_tokens, embedding_dim]
            raw_embeddings = []
            for i in sequence_indices:
                # Each sample has shape [num_tokens, embedding_dim]
                embedding = f['embeddings'][i]
                raw_embeddings.append(embedding)
            raw_embeddings = np.stack(raw_embeddings)
            
            # Aggregate tokens to get shape [sequence_length, embedding_dim]
            embeddings = self._aggregate_tokens(raw_embeddings)
            
            # Extract labels for sequence
            labels = []
            for i in sequence_indices:
                if i < len(f['labels']):
                    label = f['labels'][i]
                    labels.append(label)
                else:
                    # Use a default label if index is out of bounds
                    labels.append(0)
            labels = np.array(labels)
            
            # Create sequence-level label (majority vote)
            sequence_label = int(np.mean(labels) > 0.5)
            
            # Extract sub-labels if available
            sub_labels = None
            if 'sub_labels' in f and f['sub_labels'].shape[0] > 0:
                sub_labels = []
                for i in sequence_indices:
                    if i < len(f['sub_labels']):
                        sub_label = f['sub_labels'][i]
                        sub_labels.append(sub_label)
                    else:
                        # Use zeros as default sub-labels
                        sub_label = np.zeros(f['sub_labels'].shape[1])
                        sub_labels.append(sub_label)
                sub_labels = np.array(sub_labels)
            
            return {
                'embeddings': torch.FloatTensor(embeddings),
                'chunk_labels': torch.LongTensor(labels),
                'sequence_label': torch.LongTensor([sequence_label]),
                'sub_labels': torch.FloatTensor(sub_labels) if sub_labels is not None else None
            }


class HierarchicalDeceptionModel(nn.Module):
    """
    Simplified hierarchical model for deception detection using sequences of embeddings.
    This version is optimized for the specific embedding structure in the dataset.
    """
    
    def __init__(self, 
                 embed_dim, 
                 hidden_dim=768,  # Hidden dimension for transformer layers
                 num_transformer_layers=4,
                 num_attention_heads=8,
                 dim_feedforward=2048,
                 dropout=0.2):
        super(HierarchicalDeceptionModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Projection to reduce embedding dimension for transformer
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 100, hidden_dim))  # Max 100 sequence length
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Transformer encoder for sequence processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Chunk-level classifier
        self.chunk_classifier = nn.Linear(hidden_dim, 2)
        
        # Sequence-level classifier (for the entire sequence)
        self.sequence_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x):
        """
        Process sequence of embeddings
        
        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            chunk_logits: Logits for each chunk [batch_size, seq_len, 2]
            sequence_logits: Logits for the entire sequence [batch_size, 2]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project embeddings to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Process with transformer
        x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # Chunk-level predictions
        chunk_logits = self.chunk_classifier(x)  # [batch_size, seq_len, 2]
        
        # Sequence-level prediction (global pooling)
        global_features = x.mean(dim=1)  # [batch_size, hidden_dim]
        sequence_logits = self.sequence_classifier(global_features)  # [batch_size, 2]
        
        return chunk_logits, sequence_logits


class HierarchicalTrainer:
    """Trainer for the hierarchical deception detection model"""
    
    def __init__(self, config):
        self.config = config
        os.makedirs(config["hierarchical_models_dir"], exist_ok=True)
        
        # Initialize wandb
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            tags=config["wandb"]["tags"],
            config={k: v for k, v in config.items() if k != "wandb"},  # Exclude wandb config itself
            settings=wandb.Settings(start_method="thread")  # Use thread mode for better compatibility
        )
    
    def initialize_model(self, embed_dim):
        """Create and initialize the model"""
        print(f"Creating hierarchical model with embed_dim={embed_dim}")
        model = HierarchicalDeceptionModel(
            embed_dim=embed_dim,
            hidden_dim=768,  # Reduced dimension for transformer
            num_transformer_layers=self.config["transformer_layers"],
            num_attention_heads=self.config["transformer_heads"],
            dim_feedforward=self.config["transformer_dim_feedforward"],
            dropout=self.config["dropout"]
        )
        
        model.to(self.config["device"])
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config["num_epochs"], 
            eta_min=1e-6
        )
        
        return model, optimizer, scheduler
    
    def train_epoch(self, model, optimizer, train_loader, chunk_criterion, sequence_criterion):
        """Train for one epoch"""
        model.train()
        
        epoch_chunk_loss = 0
        epoch_sequence_loss = 0
        epoch_total_loss = 0
        
        chunk_preds = []
        chunk_labels = []
        sequence_preds = []
        sequence_labels = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Print tensor shapes for debugging the first batch
            if batch_idx == 0:
                print(f"Batch shapes:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape}")
            
            embeddings = batch['embeddings'].to(self.config["device"])
            chunk_targets = batch['chunk_labels'].to(self.config["device"])
            sequence_targets = batch['sequence_label'].to(self.config["device"]).squeeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass
            chunk_logits, sequence_logits = model(embeddings)
            
            # Reshape chunk logits and targets for loss computation
            batch_size, seq_len, num_classes = chunk_logits.shape
            chunk_logits_flat = chunk_logits.reshape(-1, num_classes)
            chunk_targets_flat = chunk_targets.reshape(-1)
            
            # Compute losses
            chunk_loss = chunk_criterion(chunk_logits_flat, chunk_targets_flat)
            sequence_loss = sequence_criterion(sequence_logits, sequence_targets)
            
            # Combined loss (weighted sum)
            total_loss = 0.3 * chunk_loss + 0.7 * sequence_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            epoch_chunk_loss += chunk_loss.item()
            epoch_sequence_loss += sequence_loss.item()
            epoch_total_loss += total_loss.item()
            
            # Save predictions for metrics
            chunk_preds.append(torch.argmax(chunk_logits, dim=2).cpu())
            chunk_labels.append(chunk_targets.cpu())
            sequence_preds.append(torch.argmax(sequence_logits, dim=1).cpu())
            sequence_labels.append(sequence_targets.cpu())
        
        # Combine predictions
        chunk_preds = torch.cat(chunk_preds, dim=0).flatten()
        chunk_labels = torch.cat(chunk_labels, dim=0).flatten()
        sequence_preds = torch.cat(sequence_preds, dim=0)
        sequence_labels = torch.cat(sequence_labels, dim=0)
        
        # Calculate metrics
        chunk_acc = accuracy_score(chunk_labels, chunk_preds)
        chunk_f1 = f1_score(chunk_labels, chunk_preds, zero_division=0)
        sequence_acc = accuracy_score(sequence_labels, sequence_preds)
        sequence_f1 = f1_score(sequence_labels, sequence_preds, zero_division=0)
        
        # Average losses
        avg_chunk_loss = epoch_chunk_loss / len(train_loader)
        avg_sequence_loss = epoch_sequence_loss / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)
        
        return {
            'chunk_loss': avg_chunk_loss,
            'sequence_loss': avg_sequence_loss,
            'total_loss': avg_total_loss,
            'chunk_acc': chunk_acc,
            'chunk_f1': chunk_f1,
            'sequence_acc': sequence_acc,
            'sequence_f1': sequence_f1
        }
    
    def validate(self, model, val_loader, chunk_criterion, sequence_criterion):
        """Validate model"""
        model.eval()
        
        val_chunk_loss = 0
        val_sequence_loss = 0
        val_total_loss = 0
        
        chunk_preds = []
        chunk_labels = []
        sequence_preds = []
        sequence_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                embeddings = batch['embeddings'].to(self.config["device"])
                chunk_targets = batch['chunk_labels'].to(self.config["device"])
                sequence_targets = batch['sequence_label'].to(self.config["device"]).squeeze(1)
                
                # Forward pass
                chunk_logits, sequence_logits = model(embeddings)
                
                # Reshape chunk logits and targets for loss computation
                batch_size, seq_len, num_classes = chunk_logits.shape
                chunk_logits_flat = chunk_logits.reshape(-1, num_classes)
                chunk_targets_flat = chunk_targets.reshape(-1)
                
                # Compute losses
                chunk_loss = chunk_criterion(chunk_logits_flat, chunk_targets_flat)
                sequence_loss = sequence_criterion(sequence_logits, sequence_targets)
                
                # Combined loss
                total_loss = 0.3 * chunk_loss + 0.7 * sequence_loss
                
                # Track metrics
                val_chunk_loss += chunk_loss.item()
                val_sequence_loss += sequence_loss.item()
                val_total_loss += total_loss.item()
                
                # Save predictions for metrics
                chunk_preds.append(torch.argmax(chunk_logits, dim=2).cpu())
                chunk_labels.append(chunk_targets.cpu())
                sequence_preds.append(torch.argmax(sequence_logits, dim=1).cpu())
                sequence_labels.append(sequence_targets.cpu())
        
        # Combine predictions
        chunk_preds = torch.cat(chunk_preds, dim=0).flatten()
        chunk_labels = torch.cat(chunk_labels, dim=0).flatten()
        sequence_preds = torch.cat(sequence_preds, dim=0)
        sequence_labels = torch.cat(sequence_labels, dim=0)
        
        # Calculate metrics
        chunk_acc = accuracy_score(chunk_labels, chunk_preds)
        chunk_f1 = f1_score(chunk_labels, chunk_preds, zero_division=0)
        sequence_acc = accuracy_score(sequence_labels, sequence_preds)
        sequence_f1 = f1_score(sequence_labels, sequence_preds, zero_division=0)
        
        # Calculate AUC for sequence predictions
        try:
            fpr, tpr, _ = roc_curve(sequence_labels, sequence_preds, pos_label=1)
            sequence_auc = auc(fpr, tpr)
        except:
            sequence_auc = 0.5  # Default if calculation fails
        
        # Average losses
        avg_chunk_loss = val_chunk_loss / len(val_loader)
        avg_sequence_loss = val_sequence_loss / len(val_loader)
        avg_total_loss = val_total_loss / len(val_loader)
        
        # Generate classification report
        report = classification_report(
            sequence_labels.numpy(), 
            sequence_preds.numpy(),
            target_names=["truth", "deception"],
            zero_division=0,
            output_dict=True
        )
        
        return {
            'chunk_loss': avg_chunk_loss,
            'sequence_loss': avg_sequence_loss,
            'total_loss': avg_total_loss,
            'chunk_acc': chunk_acc,
            'chunk_f1': chunk_f1,
            'sequence_acc': sequence_acc,
            'sequence_f1': sequence_f1,
            'sequence_auc': sequence_auc,
            'report': report
        }
    
    def train(self, model, optimizer, scheduler, train_loader, val_loader):
        """Train the model for multiple epochs"""
        # Loss functions
        chunk_criterion = nn.CrossEntropyLoss()
        sequence_criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0.0
        best_epoch = 0
        early_stop_patience = 5
        early_stop_counter = 0
        
        for epoch in range(self.config["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(model, optimizer, train_loader, chunk_criterion, sequence_criterion)
            
            # Validate
            val_metrics = self.validate(model, val_loader, chunk_criterion, sequence_criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Chunk Acc: {train_metrics['chunk_acc']:.4f}, "
                  f"Sequence Acc: {train_metrics['sequence_acc']:.4f}, "
                  f"Sequence F1: {train_metrics['sequence_f1']:.4f}")
            
            print(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Chunk Acc: {val_metrics['chunk_acc']:.4f}, "
                  f"Sequence Acc: {val_metrics['sequence_acc']:.4f}, "
                  f"Sequence F1: {val_metrics['sequence_f1']:.4f}, "
                  f"Sequence AUC: {val_metrics['sequence_auc']:.4f}")
            
            # Log to wandb
            log_dict = {
                'epoch': epoch+1,
                'lr': optimizer.param_groups[0]['lr'],
                'train/chunk_loss': train_metrics['chunk_loss'],
                'train/sequence_loss': train_metrics['sequence_loss'],
                'train/total_loss': train_metrics['total_loss'],
                'train/chunk_acc': train_metrics['chunk_acc'],
                'train/chunk_f1': train_metrics['chunk_f1'],
                'train/sequence_acc': train_metrics['sequence_acc'],
                'train/sequence_f1': train_metrics['sequence_f1'],
                'val/chunk_loss': val_metrics['chunk_loss'],
                'val/sequence_loss': val_metrics['sequence_loss'],
                'val/total_loss': val_metrics['total_loss'],
                'val/chunk_acc': val_metrics['chunk_acc'],
                'val/chunk_f1': val_metrics['chunk_f1'],
                'val/sequence_acc': val_metrics['sequence_acc'],
                'val/sequence_f1': val_metrics['sequence_f1'],
                'val/sequence_auc': val_metrics['sequence_auc'],
            }
            
            # Add detailed metrics from classification report
            for target in ['truth', 'deception']:
                for metric in ['precision', 'recall', 'f1-score']:
                    key = f'val/{target}_{metric}'
                    value = val_metrics['report'][target][metric]
                    log_dict[key] = value
            
            wandb.log(log_dict)
            
            # Save best model
            if val_metrics['sequence_f1'] > best_val_f1:
                best_val_f1 = val_metrics['sequence_f1']
                best_epoch = epoch + 1
                
                # Save model
                model_path = os.path.join(
                    self.config["hierarchical_models_dir"], 
                    f"best_model_seq{self.config['sequence_length']}_ep{epoch+1}.pt"
                )
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved to {model_path}")
                
                # Reset early stopping counter
                early_stop_counter = 0
            else:
                # Increment early stopping counter if validation accuracy doesn't improve
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered! No improvement for {early_stop_patience} epochs.")
                    print(f"Stopping at epoch {epoch+1}. Best results were at epoch {best_epoch}.")
                    break
        
        print(f"\nTraining completed. Best model from epoch {best_epoch} with sequence F1: {best_val_f1:.4f}")
        wandb.run.summary["best_val_f1"] = best_val_f1
        wandb.run.summary["best_epoch"] = best_epoch
        
        return best_epoch, best_val_f1
    
    def find_embeddings_file(self, split, train_file, test_file):
        """Find embeddings file with more flexible matching"""
        embedding_dir = self.config["embeddings_dir"]
        print(f"Searching for {split} embeddings in {embedding_dir}")
        
        # List all files in the embeddings directory
        all_files = os.listdir(embedding_dir)
        
        # First, try exact match
        pattern = f"{split}_{train_file}_{test_file}.h5"
        if pattern in all_files:
            return os.path.join(embedding_dir, pattern)
        
        # Next, try with .csv extensions
        pattern_with_csv = f"{split}_{train_file}.csv_{test_file}.csv.h5"
        if pattern_with_csv in all_files:
            return os.path.join(embedding_dir, pattern_with_csv)
            
        # Try with only the file prefix (without extension)
        train_prefix = train_file.split('.')[0]
        test_prefix = test_file.split('.')[0]
        for filename in all_files:
            if filename.startswith(f"{split}_") and train_prefix in filename and test_prefix in filename and filename.endswith(".h5"):
                return os.path.join(embedding_dir, filename)
        
        # Try a glob pattern
        glob_pattern = os.path.join(embedding_dir, f"{split}_*{train_prefix}*{test_prefix}*.h5")
        matches = glob.glob(glob_pattern)
        if matches:
            return matches[0]
            
        # One last attempt: just look for any file with both train and test names
        for filename in all_files:
            if filename.endswith(".h5") and (train_file in filename or train_prefix in filename) and (test_file in filename or test_prefix in filename) and split in filename:
                return os.path.join(embedding_dir, filename)
                
        # If we got here, no file was found
        return None
    
    def check_file_structure(self, filepath):
        """Check and print the structure of an H5 file for debugging"""
        print(f"Checking file structure of {filepath}")
        try:
            with h5py.File(filepath, "r") as f:
                # List all top-level items
                print("Top level keys:", list(f.keys()))
                
                # Check embeddings
                if 'embeddings' in f:
                    print(f"Embeddings shape: {f['embeddings'].shape}")
                    print(f"Embeddings dtype: {f['embeddings'].dtype}")
                    # Sample some values
                    print(f"First embedding shape: {f['embeddings'][0].shape}")
                
                # Check labels
                if 'labels' in f:
                    print(f"Labels shape: {f['labels'].shape}")
                    print(f"Labels dtype: {f['labels'].dtype}")
                    label_counts = np.bincount(f['labels'][:])
                    print(f"Label distribution: {label_counts}")
                    print(f"Label percentages: {label_counts / np.sum(label_counts) * 100:.2f}%")
                
                # Check attributes
                print("File attributes:", dict(f.attrs.items()))
                
                # Check raw features if present
                if 'features' in f:
                    print("Features keys:", list(f['features'].keys()))
                    for key in f['features']:
                        print(f"Feature '{key}' shape: {f['features'][key].shape}")
                        print(f"Feature '{key}' dtype: {f['features'][key].dtype}")
        except Exception as e:
            print(f"Error checking file: {e}")
    
    def process_protocol(self, protocol):
        """Process a single protocol"""
        train_file, test_file = protocol
        protocol_key = f"{train_file}_{test_file}"
        
        print(f"\nProcessing protocol: {train_file} (train), {test_file} (test)")
        
        # Find embeddings files using more flexible matching
        train_embeddings_path = self.find_embeddings_file("train", train_file, test_file)
        test_embeddings_path = self.find_embeddings_file("test", train_file, test_file)
        
        # Check if files were found
        if train_embeddings_path is None:
            print(f"No train embeddings found for protocol {protocol_key}")
            return None
            
        if test_embeddings_path is None:
            print(f"No test embeddings found for protocol {protocol_key}")
            return None
            
        print(f"Found embeddings files:\n  Train: {train_embeddings_path}\n  Test: {test_embeddings_path}")
        
        # Check file structure for debugging
        self.check_file_structure(train_embeddings_path)
        
        # Get embedding dimension directly from the file
        with h5py.File(train_embeddings_path, 'r') as f:
            embed_dim = f['embeddings'].shape[2]
            print(f"Using embedding dimension: {embed_dim}")
        
        # Create datasets with appropriate token aggregation
        try:
            train_dataset = SequenceDataset(
                train_embeddings_path,
                sequence_length=self.config["sequence_length"],
                token_aggregation='mean',  # Using mean aggregation for tokens
                mode='random'  # For training, use random sequences
            )
            
            test_dataset = SequenceDataset(
                test_embeddings_path,
                sequence_length=self.config["sequence_length"],
                token_aggregation='mean',  # Using mean aggregation for tokens
                mode='consecutive'  # For testing, use consecutive sequences
            )
            
            print(f"Created datasets with {len(train_dataset)} training sequences and {len(test_dataset)} test sequences")
            
            # Check a sample from dataset
            sample = train_dataset[0]
            print("Sample embeddings shape:", sample['embeddings'].shape)
            print("Sample chunk labels shape:", sample['chunk_labels'].shape)
            print("Sample sequence label:", sample['sequence_label'])
            
        except Exception as e:
            print(f"Error creating datasets: {e}")
            return None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            drop_last=True  # Drop last incomplete batch
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
            drop_last=False  # Keep all test samples
        )
        
        # Initialize model with correct embedding dimension
        model, optimizer, scheduler = self.initialize_model(embed_dim)
        
        # Train model
        best_epoch, best_f1 = self.train(model, optimizer, scheduler, train_loader, test_loader)
        
        # Return best results
        return {
            'protocol': protocol_key,
            'best_epoch': best_epoch,
            'best_f1': best_f1
        }
    
    def process_all_protocols(self):
        """Process all protocols"""
        results = {}
        
        for protocol in self.config["protocols"]:
            # Process protocol
            result = self.process_protocol(protocol)
            if result:
                results[result['protocol']] = {
                    'best_epoch': result['best_epoch'],
                    'best_f1': result['best_f1']
                }
        
        # Save results
        results_path = os.path.join(
            self.config["hierarchical_models_dir"], 
            f"results_seq{self.config['sequence_length']}_{int(time.time())}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        return results


if __name__ == "__main__":
    # Initialize trainer
    trainer = HierarchicalTrainer(config)
    
    # Process all protocols
    results = trainer.process_all_protocols()
    
    # Print summary of results
    print("\nSummary of results:")
    for protocol, result in results.items():
        print(f"Protocol {protocol}: Best F1 = {result['best_f1']:.4f} (Epoch {result['best_epoch']})")
    
    # Finish wandb run
    wandb.finish()