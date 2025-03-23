# modality_join/cross_attention.py

import torch
import torch.nn as nn
from torch.functional import F

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal fusion module that merges a list of latent representations using
    multi-head cross-attention and an attention-based modality merge.
    Assumes that each latent tensor has shape (batch, seq=64, embed=768).

    Args:
        embed_dim (int): Dimensionality of the latent features (e.g., 768).
        num_heads (int): Number of attention heads.
        num_modalities (int): Number of modalities being fused.
        dropout (float): Dropout rate used in the attention module.
    """
    def __init__(self, embed_dim, num_heads, num_modalities, num_layers, dropout=0.15, seq_length=64):
        super(CrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Learnable modality embeddings: one per modality.
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim))
        
        self.pre_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])
        self.post_norm = nn.LayerNorm(embed_dim)

        self.attn_blocks = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(num_modalities)])
        
        # Learnable query for merging modalities (attention-based merge)
        self.merge_query = nn.Parameter(torch.randn(embed_dim))

        self.gate_layer = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_modalities)])

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
        
        # Add modality-specific embeddings and pre-normalize.
        for i, x in enumerate(latent_list):
        #     # Add modality embedding (broadcasted over the sequence)
        #     mod_embed = self.modality_embeddings[i].unsqueeze(0).unsqueeze(1).expand(B, seq, self.embed_dim)
        #     x = x + mod_embed
            current_length = x.size(1)
            if current_length != 64:
                raise Exception(f"Invalid seq length - #{i} modality - {x.size()}")
            x = self.pre_norms[i](x)
            modality_outputs.append(x)
            
        
        # Concatenate latent tokens from all modalities along the sequence dimension.
        # Shape: (B, seq * num_modalities, embed_dim)
        all_latents = torch.cat(modality_outputs, dim=1)
        # all_latents = self.pre_norm(all_latents)
        
        fused_modalities = []
        for i, x in enumerate(modality_outputs):
            # For each modality, perform cross-attention.
            # Query: x (B, seq, embed_dim); Key/Value: all_latents (B, seq*num_modalities, embed_dim)
            attn_out, _ = self.attn_blocks[i](query=x, key=all_latents, value=all_latents)
            
            gate_input = x.mean(dim=1)  # shape: (B, embed_dim)
            gate_weight = torch.sigmoid(self.gate_layer[i](gate_input))  # shape: (B, 1)
            attn_out = gate_weight.unsqueeze(1) * attn_out

            fused_modalities.append(attn_out)
        
        # Stack outputs from all modalities along a new dimension.
        # Shape becomes (B, seq, num_modalities, embed_dim)
        fused_out = torch.concat(fused_modalities, dim=2)
        fused_out = self.transformer(fused_out)
        
        return fused_out
