# modality_join/improved_cross_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedCrossModalAttentionFusion(nn.Module):
    """
    Enhanced cross-modal fusion module that dynamically weighs modalities and 
    implements stronger normalization and residual connections.
    
    Improvements:
    1. Modality-specific importance weighting
    2. Enhanced layer normalization
    3. Stronger residual connections
    4. Attention dropout for regularization
    5. Gated integration of modalities
    """
    def __init__(self, embed_dim, num_heads, num_modalities, num_layers, dropout=0.15, seq_length=64):
        super(ImprovedCrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Modality embeddings with positional info
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim))
        
        # Improved pre and post normalization
        self.pre_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])
        self.post_norm = nn.LayerNorm(embed_dim)
        
        # Dynamic modality weighting network
        self.modality_weight_net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # Cross-attention blocks with increased dropout
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout+0.05, batch_first=True) 
            for _ in range(num_modalities)
        ])
        
        # Gated fusion with context
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Context pooling (for computing global context)
        self.context_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Final transformer for joint processing
        transformer_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim*num_modalities,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layers, num_layers=num_layers)
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(embed_dim*num_modalities, embed_dim*num_modalities),
            nn.LayerNorm(embed_dim*num_modalities),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.seq_length = seq_length
    
    def forward(self, latent_list):
        """
        Args:
            latent_list (list[Tensor]): List of latent representations.
                Each tensor has shape (B, seq, embed_dim), e.g. (B, 64, 768).
                
        Returns:
            Tensor: Fused latent representation with enhanced cross-modal information.
        """
        B, seq, _ = latent_list[0].shape
        
        # Add modality-specific embeddings and normalize
        modality_outputs = []
        for i, x in enumerate(latent_list):
            # Ensure correct sequence length
            current_length = x.size(1)
            if current_length != self.seq_length:
                raise ValueError(f"Invalid seq length - #{i} modality - {x.size()}")
                
            # Apply pre-normalization
            x = self.pre_norms[i](x)
            
            # Add modality embedding (broadcasted over the sequence)
            mod_embed = self.modality_embeddings[i].unsqueeze(0).unsqueeze(1).expand(B, seq, self.embed_dim)
            x = x + mod_embed * 0.1  # Scale down the embedding impact
            
            modality_outputs.append(x)
        
        # Compute modality importance weights
        importance_weights = []
        for i, x in enumerate(modality_outputs):
            # Global pooling for modality representation
            mod_pooled = torch.mean(x, dim=1)  # (B, embed_dim)
            weight = self.modality_weight_net(mod_pooled)  # (B, 1)
            importance_weights.append(F.softplus(weight))  # Use softplus for non-negative weights
        
        # Normalize importance weights
        importance_weights = torch.cat(importance_weights, dim=1)  # (B, num_modalities)
        importance_weights = F.softmax(importance_weights, dim=1)  # (B, num_modalities)
        
        # Compute global context for each modality
        all_latents = torch.cat(modality_outputs, dim=1)  # (B, seq*num_modalities, embed_dim)
        global_context = torch.mean(all_latents, dim=1)  # (B, embed_dim)
        global_context = self.context_pool(global_context)  # (B, embed_dim)
        
        # Enhanced cross-attention with gating
        fused_modalities = []
        for i, x in enumerate(modality_outputs):
            # Cross-attention between this modality and all others
            attn_out, _ = self.attn_blocks[i](
                query=x, 
                key=all_latents, 
                value=all_latents
            )
            
            # Gated integration with global context
            context_expanded = global_context.unsqueeze(1).expand_as(x)  # (B, seq, embed_dim)
            context_concat = torch.cat([x, context_expanded], dim=2)  # (B, seq, embed_dim*2)
            gate = self.gate_layers[i](context_concat)  # (B, seq, embed_dim)
            
            # Apply attention with importance weighting and gating
            mod_weight = importance_weights[:, i].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            gated_output = gate * attn_out + (1 - gate) * x  # Residual connection with gate
            weighted_output = mod_weight * gated_output  # Apply modality importance
            
            fused_modalities.append(weighted_output)
        
        # Concatenate and apply final transformer
        fused_out = torch.cat(fused_modalities, dim=2)  # (B, seq, embed_dim*num_modalities)
        fused_out = self.transformer(fused_out)  # Apply joint reasoning
        
        # Final projection with residual connection
        output = self.final_projection(fused_out) + fused_out  # Residual connection
        
        return output