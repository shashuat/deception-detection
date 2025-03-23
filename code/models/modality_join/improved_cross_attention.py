# modality_join/improved_cross_attention_simple.py

import torch
import torch.nn as nn
from torch.functional import F

class ImprovedCrossModalAttentionFusion(nn.Module):
    """
    Improved cross-modal fusion module with minimal changes to the original.
    Adds layer normalization and improved attention weights.
    """
    def __init__(self, embed_dim, num_heads, num_modalities, num_layers, dropout=0.15, seq_length=64):
        super(ImprovedCrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Same as original, but with improved initialization
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim) * 0.02)
        
        # Add layer normalization before and after attention
        self.pre_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])
        self.post_norm = nn.LayerNorm(embed_dim)

        # Keep the original attention blocks but with slightly higher dropout
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout+0.05, batch_first=True) 
            for _ in range(num_modalities)
        ])
        
        # Keep original gate layer
        self.gate_layer = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_modalities)])
        
        # Keep original transformer
        transformer_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim*num_modalities,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layers, num_layers=num_layers)
        self.seq_length = seq_length
    
    def forward(self, latent_list):
        """
        Args:
            latent_list (list[Tensor]): List of latent representations.
                Each tensor has shape (B, seq, embed_dim), e.g. (B, 64, 768).
                
        Returns:
            Tensor: Fused latent representation of shape (B, seq, embed_dim).
        """
        B, seq, _ = latent_list[0].shape
        modality_outputs = []
        
        # Apply normalization to each modality
        for i, x in enumerate(latent_list):
            current_length = x.size(1)
            if current_length != self.seq_length:
                raise ValueError(f"Invalid seq length - #{i} modality - {x.size()}")
            
            # Apply pre-normalization
            x = self.pre_norms[i](x)
            modality_outputs.append(x)
        
        # Concatenate latent tokens from all modalities (same as original)
        all_latents = torch.cat(modality_outputs, dim=1)
        
        # Apply cross-attention for each modality (similar to original)
        fused_modalities = []
        for i, x in enumerate(modality_outputs):
            # Cross-attention
            attn_out, _ = self.attn_blocks[i](query=x, key=all_latents, value=all_latents)
            
            # Apply gating (same as original)
            gate_input = x.mean(dim=1)  # (B, embed_dim)
            gate_weight = torch.sigmoid(self.gate_layer[i](gate_input))  # (B, 1)
            attn_out = gate_weight.unsqueeze(1) * attn_out
            
            # Add residual connection (improvement)
            attn_out = attn_out + x
            
            fused_modalities.append(attn_out)
        
        # Stack the outputs (same as original)
        fused_out = torch.cat(fused_modalities, dim=2)
        
        # Apply transformer (same as original)
        fused_out = self.transformer(fused_out)
        
        # Apply final normalization (improvement)
        fused_out = self.post_norm(fused_out)
        
        return fused_out