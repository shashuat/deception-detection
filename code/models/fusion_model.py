# fusion_model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.encoders.audio_encoder import WavEncoder, WhisperTokenEncoder
from models.encoders.text_encoder import TextEncoder
from models.encoders.visual_encoder import FaceEncoder
from models.encoders.face_encoder import CombinedFace3DViT

from models.modality_join.cross_attention import CrossModalAttentionFusion
from models.modality_join.cross_fusion import CrossFusionModule

ENCODER_DIM = 768
ENCODER_SEQ_LENGTH = 256

class MultiTaskClassifier(nn.Module):
    def __init__(self, encoder_dim, num_sub_labels):
        super(MultiTaskClassifier, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.LayerNorm(encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(encoder_dim // 2, encoder_dim // 4),
            nn.LayerNorm(encoder_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.main_head = nn.Linear(encoder_dim // 4, 2)
        self.sub_heads = nn.ModuleList([nn.Linear(encoder_dim // 4, 2) for _ in range(num_sub_labels)])
    
    def forward(self, x):
        # x shape: (B, seq, encoder_dim)
        # Pool out the seq dimension if you only need a single vector representation
        x = x.mean(dim=1)   # (B, encoder_dim)
        x = self.shared_layers(x)  # (B, encoder_dim // 4)

        main_out = self.main_head(x)  # (B, 2)

        # Produce sub-label outputs as (B, num_sub_labels, 2)
        sub_outs = []
        for head in self.sub_heads:
            sub_out = head(x)  # (B, 2)
            sub_outs.append(sub_out)
        sub_outs = torch.stack(sub_outs, dim=1)  # (B, num_sub_labels, 2)

        return main_out, sub_outs

class Fusion(nn.Module):
    """
    Multimodal fusion model that combines audio and visual features
    with different fusion strategies and adapter options.
    """

    ALLOWED_MODALITIES = ["text", "audio", "whisper", "faces", "faces-adv"]

    def __init__(self, fusion_type, modalities, num_layers, adapter, adapter_type, multi=False, num_sub_labels=0):
        super(Fusion, self).__init__()

        assert fusion_type in ["concat", "cross2", "cross_attention"], "The fusion type should be 'concat' or 'cross2' or 'corss_attention'"
        assert all(modality in Fusion.ALLOWED_MODALITIES for modality in modalities), f"The possible modalities are: {Fusion.ALLOWED_MODALITIES}"
        
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        self.modalities = modalities
        self.multi = multi  # multitask learning with multiple losses

        self.encoders = nn.ModuleDict({
            "audio": WavEncoder(num_layers, adapter, adapter_type) if "audio" in self.modalities else None,
            "whisper": WhisperTokenEncoder(ENCODER_DIM, num_layers=num_layers, num_heads=6, hidden_dim=1024) if "whisper" in self.modalities else None,
            "text": TextEncoder(embedding_dim=ENCODER_DIM, transformer_layers=num_layers) if "text" in self.modalities else None,
            "faces": FaceEncoder(num_layers, adapter, adapter_type) if "faces" in self.modalities else None,
            "faces-adv": CombinedFace3DViT(ENCODER_DIM, num_layers=num_layers) if "faces-adv" in self.modalities else None
        })

        self.encoders_keys = {k for k, v in self.encoders.items() if v is not None}

        # Fusion and classification components
        if self.fusion_type == "concat":
            self.classifier = nn.Sequential(nn.Linear(768 * len(modalities), 2))

        elif self.fusion_type == "cross2": # Cross-modal fusion
            cross_conv_layer = []
            for i in range(self.num_layers):
                cross_conv_layer.append(
                    CrossFusionModule(encoder_dim=ENCODER_DIM, dim=256, num_modalities=len(modalities))
                )
            self.cross_conv_layer = nn.ModuleList(cross_conv_layer)
            
            # Main classifier
            self.classifier = nn.Sequential(
                nn.Linear(64 * self.num_layers, 64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 2)
            )
        
        elif self.fusion_type == "cross_attention":
            self.cross_attention = CrossModalAttentionFusion(ENCODER_DIM, 12, num_modalities=len(modalities), num_layers=num_layers, dropout=0.1)
            self.classifier = MultiTaskClassifier(ENCODER_DIM*len(modalities), num_sub_labels)

        self.multi_classifier = {}
        if self.multi:
            self.multi_classifier = nn.ModuleDict({
                k: nn.Linear(ENCODER_DIM, 2) if v is not None else None
                for k, v in self.encoders.items()
                if v is not None
            })

    def to_device(self, device):
        self.to(device)
        for v in self.encoders.values():
            if v: v.to(device)

        for v in self.multi_classifier.values():
            if v: v.to(device)

    def forward(self, inputs):
        for k in inputs:
            if k not in self.ALLOWED_MODALITIES: raise Exception("Modality unknow or not managed by the model (see modalities config)")
        
        if self.fusion_type == "concat":
            latents = []

            for mod in self.encoders_keys:
                input_mod = mod.split('-')[0]
                latents.append(self.encoders[mod](inputs[input_mod], False))

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

            # stacked = []
            # for i, latent in enumerate(zip(*latents)):
            #     stacked.append(self.corss_attention_layers[i](latent))
            
            # stacked = torch.stack(stacked) # (num_layers, B, seq, 768)
            # num_layers, B, seq, embed_dim = stacked.shape
            # stacked = stacked.view(B, num_layers * seq, embed_dim)  # (B, num_layers*seq, embed_dim)
            # fused_output, _ = self.attn(stacked, stacked, stacked)

        else:
            raise ValueError(f"Undefined fusion type: {self.fusion_type}")

        # classification
        logits, sub_labels_logits = self.classifier(fused_output) # (batch, embed)

        # multitask learning
        sub_outputs = {}
        if self.multi:
            for k, classifier in self.multi_classifier.items():
                sub_outputs[k] = torch.mean(classifier(latents_dict[k]), 1)
            return logits, sub_labels_logits, sub_outputs #torch.mean(logits, 1)
        return logits, sub_labels_logits, None
