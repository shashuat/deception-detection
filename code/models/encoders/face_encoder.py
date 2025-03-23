import math
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F

from models.modality_join.cross_attention import CrossModalAttentionFusion
from models.encoders.color_encoder import ColorVariation3DViT

def multi_scale_branch(x, scale_factor=0.5):
    # x: (B, C, T, H, W)
    B, C, T, H, W = x.shape
    new_H = max(1, int(H * scale_factor))
    new_W = max(1, int(W * scale_factor))
    pooled = F.adaptive_avg_pool3d(x, output_size=(T, new_H, new_W))
    return pooled

def get_sinusoidal_positional_encoding(seq_len, embed_dim, device):
    """Compute sinusoidal positional encodings of shape (1, seq_len, embed_dim)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(seq_len, embed_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class VideoViT(nn.Module):
    def __init__(self, in_channels=3, img_size=160, seq_len=64,
                 patch_size=16, emb_dim=128, num_heads=4, depth=6):
        """
        in_channels: Number of channels (e.g., 3 for RGB)
        img_size: Spatial resolution (assumed square, e.g., 160)
        seq_len: Number of frames in the video (e.g., 64)
        patch_size: Size of the square spatial patch (e.g., 16)
        emb_dim: Latent embedding dimension
        num_heads: Number of attention heads in the transformer
        depth: Number of transformer encoder layers
        """
        super().__init__()
        self.patch_size = patch_size
        
        # Number of patches per frame is computed from img_size and patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Each patch is flattened from (in_channels, patch_size, patch_size)
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Linear projection from flattened patch to latent embedding
        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)
        
        # CLS token for each frame
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # Learned positional embeddings for each patch in a frame
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        
        # Transformer encoder: processes a sequence of tokens per frame
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    
    def forward(self, x):
        # x shape: (B, seq_len, C, H, W)
        B, seq_len, C, H, W = x.shape
        p = self.patch_size
        
        # Process each frame independently by reshaping into (B*seq_len, C, H, W)
        x = x.reshape(B * seq_len, C, H, W)
        
        # Extract non-overlapping spatial patches:
        # x.unfold(2, p, p).unfold(3, p, p) yields shape (B*seq_len, C, H//p, p, W//p, p)
        x = x.unfold(2, p, p).unfold(3, p, p)
        # Permute to arrange patches as (B*seq_len, H//p, W//p, C, p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Flatten the spatial grid into one dimension: (B*seq_len, num_patches, C*p*p)
        x = x.reshape(x.shape[0], -1, self.patch_dim)
        
        # Project each flattened patch to the latent embedding space
        x = self.patch_embed(x)  # (B*seq_len, num_patches, emb_dim)
        
        # Add learned positional embeddings to each patch token
        x = x + self.pos_embed  # (B*seq_len, num_patches, emb_dim)
        
        # Prepend a CLS token to represent the entire frame
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B*seq_len, 1, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B*seq_len, 1+num_patches, emb_dim)
        
        # Process each frame's token sequence with the transformer
        x = self.transformer(x)  # (B*seq_len, 1+num_patches, emb_dim)
        
        # Extract the CLS token as the frame-level embedding
        x = x[:, 0, :]  # (B*seq_len, emb_dim)
        
        # Reshape back to (B, seq_len, emb_dim)
        x = x.reshape(B, seq_len, -1)
        return x


class CombinedFace3DViT(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, with_color_encoder=True, dropout=0.2):
        super().__init__()
        # Build a list of modality encoders.
        encoders = [
            VideoViT(emb_dim=embed_dim, num_heads=num_heads)
            for _ in range(1)
        ]
        if with_color_encoder:
            encoders.append(ColorVariation3DViT(embed_dim, num_heads))
        
        self.encoders = nn.ModuleList(encoders)
        self.dropout = nn.Dropout(dropout)
        # Cross-modal fusion: here we assume fusion module works on a sequence of modality tokens.
        self.fusion = CrossModalAttentionFusion(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, num_modalities=len(self.encoders))
        # Projection that takes concatenated modality representations.
        self.projection = nn.Linear(embed_dim * len(self.encoders), embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_all_layers=False):
        # Each encoder returns a single vector per sample.
        latents = [encoder(x) for encoder in self.encoders]  # list of (B, embed_dim)
        # Stack to shape (B, num_modalities, seq, embed_dim)
        # Fuse modality representations via cross-modal attention.
        latents = self.fusion(latents)  
        latents = self.dropout(latents)
        # Flatten modalities for projection.
        latents = self.projection(latents)
        return self.layer_norm(latents)
