# models/improved_fusion_model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.encoders.audio_encoder import WavEncoder, WhisperTokenEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.visual_encoder import FaceEncoder
from models.encoders.face_encoder import CombinedFace3DViT

# Import the improved cross attention module
from models.modality_join.improved_cross_attention import ImprovedCrossModalAttentionFusion
from models.modality_join.cross_fusion import CrossFusionModule

ENCODER_DIM = 768
ENCODER_SEQ_LENGTH = 64  # Changed to match the expected sequence length

class EnhancedMultiTaskClassifier(nn.Module):
    """
    Enhanced multi-task classifier with better regularization and feature extraction.
    """
    def __init__(self, encoder_dim, num_sub_labels, dropout=0.3):
        super(EnhancedMultiTaskClassifier, self).__init__()
        
        # Enhanced shared backbone with residual connections
        self.shared_layers = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.LayerNorm(encoder_dim // 2),
            nn.GELU(),  # GELU often performs better than ReLU
            nn.Dropout(dropout),
            
            nn.Linear(encoder_dim // 2, encoder_dim // 4),
            nn.LayerNorm(encoder_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Attention pooling for sequence-to-single vector reduction
        self.attention_pool = nn.Sequential(
            nn.Linear(encoder_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Main classification head with better initialization
        self.main_head = nn.Linear(encoder_dim // 4, 2)
        torch.nn.init.xavier_uniform_(self.main_head.weight)
        
        # Additional sub-class heads with xavier initialization
        self.sub_heads = nn.ModuleList([nn.Linear(encoder_dim // 4, 2) for _ in range(num_sub_labels)])
        for head in self.sub_heads:
            torch.nn.init.xavier_uniform_(head.weight)
        
    def forward(self, x):
        # x is the fused representation from all modality encoders: (B, seq, embed_dim)
        features = self.shared_layers(x)  # (B, seq, embed_dim//4)
        
        # Compute attention weights
        attn_weights = self.attention_pool(features)  # (B, seq, 1)
        
        # Apply attention pooling for main output
        pooled_features = torch.sum(features * attn_weights, dim=1)  # (B, embed_dim//4)
        main_out = self.main_head(pooled_features)  # (B, 2)
        
        # Generate outputs for sub-tasks using the same pooled features
        sub_outs = torch.stack([head(pooled_features) for head in self.sub_heads])  # (num_sub_labels, B, 2)
        
        return main_out, sub_outs

class ImprovedFusion(nn.Module):
    """
    Enhanced multimodal fusion model with improved attention mechanisms
    and better integration of modalities.
    """
    ALLOWED_MODALITIES = ["text", "audio", "whisper", "faces", "faces-adv"]

    def __init__(self, fusion_type, modalities, num_layers, adapter, adapter_type, 
                 multi=False, num_sub_labels=0, dropout=0.2):
        super(ImprovedFusion, self).__init__()

        assert fusion_type in ["concat", "cross2", "cross_attention"], \
            "The fusion type should be 'concat' or 'cross2' or 'cross_attention'"
        assert all(modality in ImprovedFusion.ALLOWED_MODALITIES for modality in modalities), \
            f"The possible modalities are: {ImprovedFusion.ALLOWED_MODALITIES}"
        
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        self.modalities = modalities
        self.multi = multi  # multitask learning with multiple losses

        # Initialize encoders with consistent interface
        self.encoders = nn.ModuleDict({
            "audio": WavEncoder(num_layers, adapter, adapter_type) if "audio" in self.modalities else None,
            "whisper": WhisperTokenEncoder(ENCODER_DIM, num_layers=num_layers, num_heads=6, hidden_dim=1024) 
                      if "whisper" in self.modalities else None,
            "text": TextEncoder(embedding_dim=ENCODER_DIM, transformer_layers=num_layers) 
                   if "text" in self.modalities else None,
            "faces": FaceEncoder(num_layers, adapter, adapter_type) if "faces" in self.modalities else None,
            "faces-adv": CombinedFace3DViT(ENCODER_DIM, num_layers=num_layers) 
                        if "faces-adv" in self.modalities else None
        })

        self.encoders_keys = {k for k, v in self.encoders.items() if v is not None}

        # Improved fusion components based on fusion type
        if self.fusion_type == "concat":
            # Enhanced concat fusion with normalization and projection
            self.modality_norms = nn.ModuleDict({
                k: nn.LayerNorm(ENCODER_DIM) for k in self.encoders_keys
            })
            self.classifier = nn.Sequential(
                nn.Linear(ENCODER_DIM * len(modalities), ENCODER_DIM),
                nn.LayerNorm(ENCODER_DIM),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ENCODER_DIM, 2)
            )

        elif self.fusion_type == "cross2":
            # Improved cross-modal fusion with better layer handling
            cross_conv_layer = []
            for i in range(self.num_layers):
                cross_conv_layer.append(
                    CrossFusionModule(encoder_dim=ENCODER_DIM, dim=256, num_modalities=len(modalities))
                )
            self.cross_conv_layer = nn.ModuleList(cross_conv_layer)
            
            # Enhanced classifier with dropout and normalization
            self.classifier = nn.Sequential(
                nn.LayerNorm(64 * self.num_layers),
                nn.Linear(64 * self.num_layers, 128),
                nn.ReLU(),
                nn.Dropout(dropout + 0.1),  # Slightly higher dropout for regularization
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)
            )
        
        elif self.fusion_type == "cross_attention":
            # Use the improved cross-attention fusion
            self.cross_attention = ImprovedCrossModalAttentionFusion(
                ENCODER_DIM, 12, num_modalities=len(modalities), 
                num_layers=num_layers, dropout=dropout
            )
            # Use the enhanced multi-task classifier
            self.classifier = EnhancedMultiTaskClassifier(
                ENCODER_DIM * len(modalities), num_sub_labels, dropout=dropout
            )

        # Multi-task classifiers with consistency
        self.multi_classifier = {}
        if self.multi:
            self.multi_classifier = nn.ModuleDict({
                k: nn.Sequential(
                    nn.LayerNorm(ENCODER_DIM),
                    nn.Linear(ENCODER_DIM, ENCODER_DIM // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ENCODER_DIM // 2, 2)
                ) if v is not None else None
                for k, v in self.encoders.items()
                if v is not None
            })

    def to_device(self, device):
        """Move model components to specified device"""
        self.to(device)
        for v in self.encoders.values():
            if v: v.to(device)

        for v in self.multi_classifier.values():
            if v: v.to(device)

    def forward(self, inputs):
        """
        Forward pass with improved fusion logic and consistency checks
        """
        # Validate inputs
        for k in inputs:
            if k not in self.ALLOWED_MODALITIES:
                raise ValueError(f"Modality {k} unknown or not managed by the model")
        
        if self.fusion_type == "concat":
            latents = []
            for mod in self.encoders_keys:
                input_mod = mod.split('-')[0]
                # Apply encoder and normalize output
                encoded = self.encoders[mod](inputs[input_mod], False)
                normalized = self.modality_norms[mod](encoded)
                latents.append(normalized)

            # Concatenate along feature dimension
            fused_output = torch.cat(latents, dim=-1)
            latents_dict = {k: v for k, v in zip(self.encoders_keys, latents)}

        elif self.fusion_type == "cross2":
            latents = []
            for mod in self.encoders_keys:
                input_mod = mod.split('-')[0]
                latents.append(self.encoders[mod](inputs[input_mod], True))

            # Apply cross fusion layer-by-layer
            feat_ls = []
            for i, latent in enumerate(zip(*latents)):
                fused_features = self.cross_conv_layer[i](latent)
                feat_ls.append(fused_features)

            # Store latents for multi-task learning
            latents_dict = {k: next(latent) for k, latent in zip(self.encoders_keys, latents)}

            # Concatenate features from all layers
            fused_output = torch.cat(feat_ls, dim=-1)

        elif self.fusion_type == "cross_attention":
            latents = []
            for mod in self.encoders_keys:
                input_mod = mod.split('-')[0]
                latents.append(self.encoders[mod](inputs[input_mod], False))
            
            # Apply improved cross-attention fusion
            fused_output = self.cross_attention(latents)
            latents_dict = {k: v for k, v in zip(self.encoders_keys, latents)}

        else:
            raise ValueError(f"Undefined fusion type: {self.fusion_type}")

        # Apply classification
        logits, sub_labels_logits = self.classifier(fused_output)

        # Multi-task outputs
        sub_outputs = {}
        if self.multi:
            for k, classifier in self.multi_classifier.items():
                # Apply attention pooling for consistent handling
                attn_weights = torch.softmax(
                    torch.mean(latents_dict[k], dim=-1, keepdim=True), 
                    dim=1
                )
                pooled = torch.sum(latents_dict[k] * attn_weights, dim=1)
                sub_outputs[k] = classifier(pooled)
            
            return logits, sub_labels_logits, sub_outputs
            
        return logits, sub_labels_logits, None