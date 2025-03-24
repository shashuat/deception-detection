import os
import sys
import torch
import numpy as np
import json
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from setup import get_env
from dataloader.audio_visual_dataset import DatasetDOLOS, af_collate_fn
from models.fusion_model import Fusion

ENV = get_env()
config = {
    # Paths
    "data_root": ENV["DOLOS_PATH"],
    "audio_path": ENV["AUDIO_PATH"],
    "visual_path": ENV["VISUAL_PATH"],
    "transcripts_path": ENV["TRANSCRIPTS_PATH"],
    "log_dir": ENV["LOGS_PATH"],
    "embeddings_dir": os.path.join(ENV["LOGS_PATH"], "embeddings"),
    
    # Model configuration
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 16,  # Increased batch size for faster processing
    "modalities": ["faces", "audio", "text", "whisper"],
    "num_layers": 4,
    "adapter": True,
    "adapter_type": "efficient_conv",
    "fusion_type": "cross_attention",
    "multi": True,
    "sub_labels": True, # Use sub labels for compatibility with the model
    
    # Protocols for extraction
    "protocols": [
        ["train_fold3.csv", "test_fold3.csv"],
        # Add more protocols as needed
    ],
    
    # Checkpoint paths - you'll need to set these to your best model paths
    "checkpoint_paths": {
        "train_fold3.csv_test_fold3.csv": os.path.join(ENV["LOGS_PATH"], "best_model_train_fold3_test_fold3.pt"),
        # Add more checkpoints as needed
    }
}

class LatentExtractor:
    def __init__(self, config):
        self.config = config
        os.makedirs(config["embeddings_dir"], exist_ok=True)
        print(f"Embeddings will be saved to {config['embeddings_dir']}")
        
    def extract_features_without_model(self, dataloader, split_name):
        """
        Extract raw input features without loading a pre-trained model.
        This is a fallback method when the checkpoint can't be loaded correctly.
        """
        all_features = {}
        all_labels = []
        all_video_ids = []
        all_sub_labels = []
        
        print(f"Extracting raw input features for {split_name} (without loading model)")
        
        for batch_idx, (data, labels, sub_labels) in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
            # Store all modality features separately
            for modal, tensor in data.items():
                if modal not in all_features:
                    all_features[modal] = []
                # Process and store each modality's features
                if modal in ['audio', 'faces', 'text', 'whisper']:
                    features = tensor.cpu().numpy()
                    all_features[modal].append(features)
            
            # Store labels and metadata
            all_labels.append(labels.cpu().numpy())
            
            # Extract video IDs if available
            if hasattr(dataloader.dataset, 'video_ids'):
                batch_indices = list(range(batch_idx * dataloader.batch_size, 
                                           min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))))
                batch_video_ids = [dataloader.dataset.video_ids[i] for i in batch_indices if i < len(dataloader.dataset.video_ids)]
                all_video_ids.extend(batch_video_ids)
            
            # Store sub-labels if available
            if sub_labels is not None:
                all_sub_labels.append(sub_labels.cpu().numpy())
                
            # Print progress periodically
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches...")
        
        # Concatenate all batches for each modality
        for modal in all_features:
            try:
                all_features[modal] = np.concatenate(all_features[modal], axis=0)
                print(f"  {modal} shape: {all_features[modal].shape}")
            except Exception as e:
                print(f"Error concatenating {modal} features: {e}")
                all_features[modal] = np.array([])
        
        # Concatenate all labels
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Concatenate sub-labels if available
        if all_sub_labels:
            try:
                all_sub_labels = np.concatenate(all_sub_labels, axis=0)
            except Exception as e:
                print(f"Error concatenating sub-labels: {e}")
                all_sub_labels = np.array([])
        
        return {
            "features": all_features,
            "labels": all_labels,
            "video_ids": all_video_ids,
            "sub_labels": all_sub_labels if all_sub_labels else None
        }
    
    def create_new_model(self, num_sub_labels=0):
        """Create a new instance of the model without loading weights"""
        print("Creating new model instance without loading weights")
        model = Fusion(
            self.config["fusion_type"],
            self.config["modalities"],
            self.config["num_layers"], 
            self.config["adapter"], 
            self.config["adapter_type"], 
            self.config["multi"],
            num_sub_labels
        )
        model.to_device(self.config["device"])
        model.eval()
        return model
    
    def extract_embeddings_from_new_model(self, model, dataloader, split_name):
        """Extract latent embeddings using a freshly created model instance"""
        all_embeddings = []
        all_labels = []
        all_video_ids = []
        all_sub_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, labels, sub_labels) in enumerate(tqdm(dataloader, desc=f"Extracting embeddings for {split_name}")):
                # Get inputs and labels
                model_input = {k: v.to(self.config["device"]) for k, v in data.items()}
                labels = labels.to(self.config["device"])
                
                # Get encoders output (without fusion)
                latents = []
                for mod in model.encoders_keys:
                    input_mod = mod.split('-')[0]
                    latent = model.encoders[mod](model_input[input_mod], False)
                    latents.append(latent)
                
                # Concatenate all modality latents along the embedding dimension
                # Shape: (batch_size, seq_len, embed_dim * num_modalities)
                concat_embeddings = torch.cat(latents, dim=-1)
                
                # Store embeddings, labels, and metadata
                all_embeddings.append(concat_embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                # Extract video IDs from the dataset
                if hasattr(dataloader.dataset, 'video_ids'):
                    batch_indices = list(range(batch_idx * dataloader.batch_size, 
                                           min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))))
                    batch_video_ids = [dataloader.dataset.video_ids[i] for i in batch_indices if i < len(dataloader.dataset.video_ids)]
                    all_video_ids.extend(batch_video_ids)
                
                # Store sub-labels if available
                if sub_labels is not None:
                    all_sub_labels.append(sub_labels.cpu().numpy())
                
                # Print progress periodically
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx} batches...")
        
        # Concatenate all batches
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        if all_sub_labels:
            try:
                all_sub_labels = np.concatenate(all_sub_labels, axis=0)
            except:
                all_sub_labels = np.array([])
        
        return {
            "embeddings": all_embeddings,
            "labels": all_labels,
            "video_ids": all_video_ids,
            "sub_labels": all_sub_labels if len(all_sub_labels) > 0 else None
        }
    
    def save_embeddings(self, embeddings_data, output_path):
        """Save embeddings and metadata to HDF5 file"""
        print(f"Saving embeddings to {output_path}")
        with h5py.File(output_path, 'w') as f:
            # Store embeddings
            f.create_dataset('embeddings', data=embeddings_data['embeddings'])
            f.create_dataset('labels', data=embeddings_data['labels'])
            
            # Store video IDs as strings
            if embeddings_data['video_ids']:
                dt = h5py.special_dtype(vlen=str)
                video_ids_dataset = f.create_dataset('video_ids', (len(embeddings_data['video_ids']),), dtype=dt)
                for i, vid in enumerate(embeddings_data['video_ids']):
                    video_ids_dataset[i] = vid
            
            # Store sub-labels if available
            if embeddings_data['sub_labels'] is not None:
                f.create_dataset('sub_labels', data=embeddings_data['sub_labels'])
                
            # Store metadata about shape and dimensions
            f.attrs['embedding_dim'] = embeddings_data['embeddings'].shape[-1]
            f.attrs['num_samples'] = embeddings_data['embeddings'].shape[0]
            f.attrs['timestamp'] = time.time()
            
            print(f"Saved {embeddings_data['embeddings'].shape[0]} samples with embedding dimension {embeddings_data['embeddings'].shape[-1]}")
    
    def save_raw_features(self, features_data, output_path):
        """Save raw features and metadata to HDF5 file"""
        print(f"Saving raw features to {output_path}")
        with h5py.File(output_path, 'w') as f:
            # Create a group for features
            features_group = f.create_group('features')
            
            # Store each modality's features
            for modal, features in features_data['features'].items():
                if features.size > 0:  # Only save non-empty arrays
                    features_group.create_dataset(modal, data=features)
                    print(f"  Saved {modal} features with shape {features.shape}")
            
            # Store labels
            f.create_dataset('labels', data=features_data['labels'])
            
            # Store video IDs as strings
            if features_data['video_ids']:
                dt = h5py.special_dtype(vlen=str)
                video_ids_dataset = f.create_dataset('video_ids', (len(features_data['video_ids']),), dtype=dt)
                for i, vid in enumerate(features_data['video_ids']):
                    video_ids_dataset[i] = vid
            
            # Store sub-labels if available
            if features_data['sub_labels'] is not None:
                f.create_dataset('sub_labels', data=features_data['sub_labels'])
                
            # Store metadata
            f.attrs['num_samples'] = len(features_data['labels'])
            f.attrs['timestamp'] = time.time()
            
            print(f"Saved {len(features_data['labels'])} samples with labels and metadata")
    
    def process_all_protocols(self):
        """Process all protocols defined in the config"""
        for protocol in self.config["protocols"]:
            train_file, test_file = protocol
            protocol_key = f"{train_file}_{test_file}"
            
            print(f"\nProcessing protocol: {train_file} (train), {test_file} (test)")
            
            # Get dataset paths
            train_anno = os.path.join(self.config["data_root"], "protocols", train_file)
            test_anno = os.path.join(self.config["data_root"], "protocols", test_file)
            
            # Initialize datasets
            print("Initializing datasets...")
            train_dataset = DatasetDOLOS(
                annotations_file=train_anno, 
                audio_dir=self.config["audio_path"], 
                img_dir=self.config["visual_path"],
                transcripts_dir=self.config["transcripts_path"],
                ignore_audio_errors=True
            )

            test_dataset = DatasetDOLOS(
                annotations_file=test_anno, 
                audio_dir=self.config["audio_path"], 
                img_dir=self.config["visual_path"],
                transcripts_dir=self.config["transcripts_path"],
                ignore_audio_errors=True
            )
            
            print(f"Dataset sizes: Train={len(train_dataset)}, Test={len(test_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=False,  # Keep original order for traceability
                collate_fn=af_collate_fn,
                num_workers=4
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=False,
                collate_fn=af_collate_fn,
                num_workers=4
            )
            
            try:
                # Try to create a new model without loading weights
                print("Creating new model to extract encoder outputs...")
                model = self.create_new_model(train_dataset.num_sub_labels)
                
                # Extract embeddings using the new model
                print(f"Extracting embeddings for training set ({len(train_dataset)} samples)...")
                train_embeddings = self.extract_embeddings_from_new_model(model, train_loader, "train")
                
                print(f"Extracting embeddings for test set ({len(test_dataset)} samples)...")
                test_embeddings = self.extract_embeddings_from_new_model(model, test_loader, "test")
                
                # Save embeddings
                train_output_path = os.path.join(self.config["embeddings_dir"], f"train_{protocol_key}.h5")
                test_output_path = os.path.join(self.config["embeddings_dir"], f"test_{protocol_key}.h5")
                
                print(f"Saving training embeddings to {train_output_path}")
                self.save_embeddings(train_embeddings, train_output_path)
                
                print(f"Saving test embeddings to {test_output_path}")
                self.save_embeddings(test_embeddings, test_output_path)
                
            except Exception as e:
                print(f"Error extracting embeddings with model: {e}")
                print("Falling back to extracting raw features...")
                
                # Extract raw features
                train_features = self.extract_features_without_model(train_loader, "train")
                test_features = self.extract_features_without_model(test_loader, "test")
                
                # Save raw features
                train_output_path = os.path.join(self.config["embeddings_dir"], f"raw_train_{protocol_key}.h5")
                test_output_path = os.path.join(self.config["embeddings_dir"], f"raw_test_{protocol_key}.h5")
                
                self.save_raw_features(train_features, train_output_path)
                self.save_raw_features(test_features, test_output_path)
            
            print(f"Completed processing for protocol {protocol_key}")

if __name__ == "__main__":
    extractor = LatentExtractor(config)
    extractor.process_all_protocols()
    print("Extraction complete!")