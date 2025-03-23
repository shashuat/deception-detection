# models/improved_fusion_model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.encoders.audio_encoder import WavEncoder, WhisperTokenEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.visual_encoder import FaceEncoder
from models.encoders.face_encoder import CombinedFace3DViT

# Import the improved cross attention - use this only if you add the file
# from models.modality_join.improved_cross_attention_simple import ImprovedCrossModalAttentionFusion
from models.modality_join.cross_attention import CrossModalAttentionFusion
from models.modality_join.cross_fusion import CrossFusionModule

ENCODER_DIM = 768

class ImprovedFusion(nn.Module):
    """
    Improved Fusion model with minimal changes to the original
    """
    ALLOWED_MODALITIES = ["text", "audio", "whisper", "faces", "faces-adv"]

    def __init__(self, fusion_type, modalities, num_layers, adapter, adapter_type, multi=False, num_sub_labels=0):
        super(ImprovedFusion, self).__init__()

        assert fusion_type in ["concat", "cross2", "cross_attention"], "The fusion type should be 'concat' or 'cross2' or 'corss_attention'"
        assert all(modality in ImprovedFusion.ALLOWED_MODALITIES for modality in modalities), f"The possible modalities are: {ImprovedFusion.ALLOWED_MODALITIES}"
        
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        self.modalities = modalities
        self.multi = multi  # multitask learning with multiple losses

        # Initialize encoders
        self.encoders = nn.ModuleDict({
            "audio": WavEncoder(num_layers, adapter, adapter_type) if "audio" in self.modalities else None,
            "whisper": WhisperTokenEncoder(ENCODER_DIM, num_layers=num_layers, num_heads=6, hidden_dim=1024) if "whisper" in self.modalities else None,
            "text": TextEncoder(embedding_dim=ENCODER_DIM, transformer_layers=num_layers) if "text" in self.modalities else None,
            "faces": FaceEncoder(num_layers, adapter, adapter_type) if "faces" in self.modalities else None,
            "faces-adv": CombinedFace3DViT(ENCODER_DIM, num_layers=num_layers) if "faces-adv" in self.modalities else None
        })

        self.encoders_keys = {k for k, v in self.encoders.items() if v is not None}

        # Fusion components based on fusion type - same as original
        if self.fusion_type == "concat":
            # IMPROVEMENT: Add layer normalization
            self.layer_norms = nn.ModuleDict({
                k: nn.LayerNorm(ENCODER_DIM) for k in self.encoders_keys
            })
            self.classifier = nn.Sequential(
                nn.Linear(768 * len(modalities), 512),
                nn.ReLU(),
                nn.Dropout(0.2),  # IMPROVEMENT: Add dropout
                nn.Linear(512, 2)
            )

        elif self.fusion_type == "cross2":
            cross_conv_layer = []
            for i in range(self.num_layers):
                cross_conv_layer.append(
                    CrossFusionModule(encoder_dim=ENCODER_DIM, dim=256, num_modalities=len(modalities))
                )
            self.cross_conv_layer = nn.ModuleList(cross_conv_layer)
            
            # IMPROVEMENT: Add more layers and dropout
            self.classifier = nn.Sequential(
                nn.Linear(64 * self.num_layers, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)
            )
        
        elif self.fusion_type == "cross_attention":
            # Use the original CrossModalAttentionFusion
            # If you want to use the improved version, swap the comment
            self.cross_attention = CrossModalAttentionFusion(
                ENCODER_DIM, 12, num_modalities=len(modalities), 
                num_layers=num_layers, dropout=0.2
            )
            
            # IMPROVEMENT: Enhanced classifier with more layers
            self.classifier = nn.Sequential(
                nn.LayerNorm(ENCODER_DIM * len(modalities)),
                nn.Linear(ENCODER_DIM * len(modalities), ENCODER_DIM),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(ENCODER_DIM, 2)
            )

        # Multi-task classifiers
        self.multi_classifier = {}
        if self.multi:
            self.multi_classifier = nn.ModuleDict({
                k: nn.Sequential(
                    nn.LayerNorm(ENCODER_DIM),  # IMPROVEMENT: Add layer normalization
                    nn.Linear(ENCODER_DIM, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),  # IMPROVEMENT: Add dropout
                    nn.Linear(256, 2)
                ) if v is not None else None
                for k, v in self.encoders.items()
                if v is not None
            })

    def to_device(self, device):
        """Move model components to device"""
        self.to(device)
        for v in self.encoders.values():
            if v: v.to(device)

        for v in self.multi_classifier.values():
            if v: v.to(device)

    def forward(self, inputs):
        """Forward pass - mostly same as original with minor improvements"""
        for k in inputs:
            if k not in self.ALLOWED_MODALITIES: 
                raise Exception("Modality unknown or not managed by the model")
        
        if self.fusion_type == "concat":
            latents = []
            for mod in self.encoders_keys:
                input_mod = mod.split('-')[0]
                # IMPROVEMENT: Apply layer norm
                latent = self.encoders[mod](inputs[input_mod], False)
                latent = self.layer_norms[mod](latent)
                latents.append(latent)

            fused_output = torch.cat(latents, dim=-1)
            latents_dict = {k: v for k, v in zip(self.encoders_keys, latents)}

        elif self.fusion_type == "cross2":
            latents = []
            for mod in self.encoders_keys:
                input_mod = mod.split('-')[0]
                latents.append(self.encoders[mod](inputs[input_mod], True))

            # cross fusion
            feat_ls = []
            for i, latent in enumerate(zip(*latents)):
                fused_features = self.cross_conv_layer[i](latent)
                feat_ls.append(fused_features)

            latents_dict = {k: v for k, v in zip(self.encoders_keys, latent)}

            # concatenate features from all layers
            fused_output = torch.cat(feat_ls, dim=-1)

        elif self.fusion_type == "cross_attention":
            latents = []
            for mod in self.encoders_keys:
                input_mod = mod.split('-')[0]
                latents.append(self.encoders[mod](inputs[input_mod], False))
            
            fused_output = self.cross_attention(latents)
            latents_dict = {k: v for k, v in zip(self.encoders_keys, latents)}

        else:
            raise ValueError(f"Undefined fusion type: {self.fusion_type}")

        # classification
        # IMPROVEMENT: Use pooling to reduce sequence dimension
        if self.fusion_type == "cross_attention":
            # Apply attention pooling to get a better representation
            attn_weights = torch.softmax(torch.mean(fused_output, dim=2, keepdim=True), dim=1)
            fused_output = torch.sum(fused_output * attn_weights, dim=1)
            logits = self.classifier(fused_output)
        else:
            # Original approach
            logits = self.classifier(fused_output)

        # multitask learning
        sub_outputs = {}
        if self.multi:
            for k, classifier in self.multi_classifier.items():
                # Apply pooling for multi-task classifiers too
                attn_weights = torch.softmax(torch.mean(latents_dict[k], dim=2, keepdim=True), dim=1)
                pooled = torch.sum(latents_dict[k] * attn_weights, dim=1)
                sub_outputs[k] = classifier(pooled)
            
            # Return dummy sub_labels_logits for compatibility
            sub_labels_logits = torch.zeros(len(self.modalities), len(logits), 2).to(logits.device)
            return logits, sub_labels_logits, sub_outputs
        
        # Return dummy sub_labels_logits for compatibility
        sub_labels_logits = torch.zeros(len(self.modalities), len(logits), 2).to(logits.device)
        return logits, sub_labels_logits, None