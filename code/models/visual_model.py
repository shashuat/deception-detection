import torch
import torch.nn as nn

from models.encoders.visual_encoder import FaceEncoder


class ViT_model(nn.Module):
    """Vision Transformer model for processing face frames"""
    def __init__(self, num_encoders, adapter, adapter_type):
        super(ViT_model, self).__init__()

        self.vit_encoder = FaceEncoder(num_encoders, adapter, adapter_type)
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
        )

    def forward(self, x):
        
        vit_features = self.vit_encoder(x)
        logits = self.classifier(vit_features)

        # Average predictions over all tokens
        return torch.mean(logits, 1) # strange ?