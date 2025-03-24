# modality_join/cross_attention.py

import torch
import torch.nn as nn

class LowRankTensorFusion(nn.Module):
    def __init__(self, input_dims, fusion_output_dim, rank):
        """
        Args:
            input_dims (list[int]): List of input dimensions for each modality.
            fusion_output_dim (int): Desired output dimension.
            rank (int): Rank for the low-rank fusion.
        """
        super().__init__()
        self.num_modalities = len(input_dims)
        self.rank = rank

        # Create a projection for each modality to map to a low-rank space.
        self.factors = nn.ModuleList([
            nn.Linear(dim, rank, bias=False) for dim in input_dims
        ])
        
        # Final fusion layer to map the elementwise product (of size rank) to the output dimension.
        self.fusion_weights = nn.Linear(rank, fusion_output_dim, bias=True)

    def forward(self, modality_list):
        """
        Args:
            modality_list (list[Tensor]): List of modality outputs, each of shape (B, seq, D)
        Returns:
            Tensor: Fused output of shape (B, seq, fusion_output_dim)
        """
        # Project each modality into a low-rank space: result shape (B, seq, rank) per modality.
        projected = [proj(mod) for proj, mod in zip(self.factors, modality_list)]
        
        # Multiply element-wise across modalities.
        fusion_tensor = projected[0]
        for proj in projected[1:]:
            fusion_tensor = fusion_tensor * proj  # (B, seq, rank)
        
        # Map the fused low-rank representation to the desired output dimension.
        fused_output = self.fusion_weights(fusion_tensor)  # (B, seq, fusion_output_dim)
        return fused_output

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal fusion module that merges a list of latent representations using
    multi-head cross-attention and then fuses them with low-rank tensor fusion.
    Assumes that each latent tensor has shape (B, seq=64, embed_dim).
    """
    def __init__(self, embed_dim, num_heads, num_modalities, num_layers, dropout=0.1, seq_length=64, fusion_rank=32):
        super(CrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Learnable modality embeddings: one per modality.
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim))
        
        self.pre_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])
        self.concat_norm = nn.LayerNorm(embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities)
        ])
        
        # Learnable query for merging modalities (attention-based merge) if needed.
        self.merge_query = nn.Parameter(torch.randn(embed_dim))
        self.gate_layer = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_modalities)])

        # Low-rank tensor fusion module: projects each modality (of dimension embed_dim)
        # to a low-rank space and fuses them to get an output of dimension embed_dim.
        self.lrtf = LowRankTensorFusion(
            input_dims=[embed_dim] * num_modalities,
            fusion_output_dim=embed_dim,
            rank=fusion_rank
        )

        transformer_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
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
                Each tensor has shape (B, seq, embed_dim), e.g. (B, 64, embed_dim).
        Returns:
            Tensor: Fused latent representation of shape (B, seq, embed_dim).
        """
        B, seq, _ = latent_list[0].shape
        modality_outputs = []
        
        # Add modality-specific embeddings and pre-normalize.
        for i, x in enumerate(latent_list):
            mod_embed = self.modality_embeddings[i].unsqueeze(0).unsqueeze(1).expand(B, seq, self.embed_dim)
            if x.size(1) != self.seq_length:
                raise Exception(f"Invalid seq length - modality {i}: {x.size()}")
            x = x + mod_embed
            x = self.pre_norms[i](x)
            modality_outputs.append(x)
            
        # Concatenate latent tokens from all modalities along the sequence dimension for context.
        # Shape: (B, seq * num_modalities, embed_dim)
        all_latents = torch.cat(modality_outputs, dim=1)
        all_latents = self.concat_norm(all_latents)
        
        # For each modality, perform cross-attention with all latents.
        fused_modalities = []
        for i, x in enumerate(modality_outputs):
            attn_out, _ = self.attn_blocks[i](query=x, key=all_latents, value=all_latents)
            gate_input = x.mean(dim=1)  # (B, embed_dim)
            gate_weight = torch.sigmoid(self.gate_layer[i](gate_input))  # (B, 1)
            attn_out = gate_weight.unsqueeze(1) * attn_out
            fused_modalities.append(attn_out)
        
        # Instead of concatenation along the token dimension, use low-rank tensor fusion.
        # fused_modalities is a list of tensors each of shape (B, seq, embed_dim)
        fused_out = self.lrtf(fused_modalities)  # (B, seq, embed_dim)
        
        # Optionally apply post-normalization and a transformer encoder.
        fused_out = self.post_norm(fused_out)
        fused_out = self.transformer(fused_out)
        
        return fused_out
