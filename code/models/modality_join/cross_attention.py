import torch
import torch.nn as nn

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal fusion module that merges a list of latent representations using
    multi-head cross-attention with gating mechanisms. Assumes each latent tensor
    has shape (B, seq, embed_dim), e.g., (B, 64, 768).

    Args:
        embed_dim (int): Dimensionality of the latent features (e.g., 768).
        num_heads (int): Number of attention heads.
        num_modalities (int): Number of modalities being fused.
        dropout (float): Dropout rate used in the attention module.
    """
    def __init__(self, embed_dim, num_heads, num_modalities, dropout=0.2):
        super(CrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        self.gate_linear = nn.Linear(embed_dim, embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)
        
        # Learnable modality embeddings: one per modality.
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim))
        # Learnable weights for fusing modalities instead of a simple mean.
        self.fuse_weights = nn.Parameter(torch.ones(num_modalities))
    
    def forward(self, latent_list):
        """
        Args:
            latent_list (list[Tensor]): List of latent representations.
                Each tensor has shape (B, seq, embed_dim), e.g., (B, 64, 768).
                
        Returns:
            Tensor: Fused latent representation of shape (B, seq, embed_dim).
        """
        B, seq, _ = latent_list[0].shape
        modality_outputs = []
        
        # add modality-specific embeddings (broadcasted over the sequence)
        for i, x in enumerate(latent_list):
            mod_embed = self.modality_embeddings[i].unsqueeze(0).unsqueeze(1).expand(B, seq, self.embed_dim)
            x = x + mod_embed
            x = self.pre_norm(x)
            modality_outputs.append(x)
        
        all_latents = torch.cat(modality_outputs, dim=1)  # (B, seq * num_modalities, embed_dim)
        # all_latents = self.pre_norm(all_latents)
        
        fused_modalities = []
        # cross-attention per modality.
        for x in modality_outputs:
            # Q: current modality tokens; K/V: all tokens from all modalities.
            attn_out, _ = self.attn(query=x, key=all_latents, value=all_latents)
            attn_out = self.dropout(attn_out)
            
            # gating mechanism
            summary = x.mean(dim=1)  # (B, embed_dim)
            gate = torch.sigmoid(self.gate_linear(summary))  # (B, embed_dim)
            gate = gate.unsqueeze(1).expand_as(x)  # (B, seq, embed_dim)
            
            # fuse the original tokens with the attended output.
            fused = gate * x + (1 - gate) * attn_out
            fused = self.post_norm(fused)
            fused_modalities.append(fused)
        
        # use learnable weights to fuse modalities
        fuse_weights = torch.softmax(self.fuse_weights, dim=0)  # (num_modalities,)
        fused_stack = torch.stack(fused_modalities, dim=0)  # (num_modalities, B, seq, embed_dim)
        fuse_weights = fuse_weights.view(self.num_modalities, 1, 1, 1)  # for broadcasting
        fused_out = (fused_stack * fuse_weights).sum(dim=0)  # (B, seq, embed_dim)
        
        return fused_out
