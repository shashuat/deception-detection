import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFusionModule(nn.Module):
    """
    Cross-modal fusion module: fuses a list of latent spaces using cross attention.
    The first modality in the list is used as an anchor, and cross attention is computed
    between it and each other modality. The cross-attended representations (with skip connections)
    are concatenated and projected to form the final fused representation.
    """
    def __init__(self, encoder_dim, dim: int = 256, num_modalities: int = 2):
        """
        Args:
            dim: Common dimension for projected features.
            num_modalities: Number of latent spaces to fuse (>=2). 
                            The first modality is used as the anchor.
        """
        super(CrossFusionModule, self).__init__()
        self.num_modalities = num_modalities
        
        # Create projection layers for each modality
        self.projections = nn.ModuleList([
            nn.Linear(encoder_dim, dim) for _ in range(num_modalities)
        ])
        
        # Shared learnable correlation weights (used in cross-attention)
        self.corr_weights = nn.Parameter(torch.empty(dim, dim).uniform_(-0.1, 0.1))
        
        # Bottleneck projection:
        # Input dimension = 2 * dim * (num_modalities - 1)
        # Output dimension = 64 (to match the original output shape [B, 64, 64])
        self.project_bottleneck = nn.Sequential(
            nn.Linear(2 * dim * (num_modalities - 1), 64),
            nn.LayerNorm(64, eps=1e-05),
            nn.ReLU()
        )
    
    def forward(self, latent_list):
        """
        Args:
            latent_list: 
                List of latent features, each with shape [B, 64, ENCODER_DIM].
                Its length must equal self.num_modalities.
        
        Returns:
            Fused features of shape [B, 64, 64].
        """
        if len(latent_list) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(latent_list)}")
        
        # Project each latent space to the common dimension: [B, 64, dim]
        projected = []
        for i, latent in enumerate(latent_list):
            # In case a latent is wrapped in a tuple, take the first element
            if isinstance(latent, tuple):
                latent = latent[0]
            projected.append(self.projections[i](latent))
        
        # Use the first modality as the anchor
        anchor = projected[0]  # [B, 64, dim]
        
        fused_pairs = []
        # Fuse the anchor with each additional modality using cross-attention
        for other in projected[1:]:
            # Prepare 'other' for matrix multiplication: [B, dim, 64]
            other_t = other.transpose(1, 2)
            
            # Compute the correlation matrix:
            # (anchor * corr_weights) @ other^T  ==> [B, 64, 64]
            a1 = torch.matmul(anchor, self.corr_weights)  # [B, 64, dim]
            cc_mat = torch.bmm(a1, other_t)  # [B, 64, 64]
            
            # Compute attention weights (note: check dim selection if needed)
            anchor_att = F.softmax(cc_mat, dim=1)           # [B, 64, 64]
            other_att  = F.softmax(cc_mat.transpose(1, 2), dim=1)  # [B, 64, 64]
            
            # Apply attention:
            # For the anchor: (B, dim, 64) = (B, 64, dim)^T @ anchor_att
            attended_anchor = torch.bmm(anchor.transpose(1, 2), anchor_att)  # [B, dim, 64]
            attended_other  = torch.bmm(other.transpose(1, 2), other_att)     # [B, dim, 64]
            
            # Add skip connections
            attended_anchor = attended_anchor + anchor.transpose(1, 2)  # [B, dim, 64]
            attended_other  = attended_other + other_t  # [B, dim, 64]
            
            # Concatenate the two branches along the channel dimension: [B, 2*dim, 64]
            fused_pair = torch.cat((attended_anchor, attended_other), dim=1)
            fused_pairs.append(fused_pair)
        
        # Concatenate all fused pairs along the channel dimension.
        # For (num_modalities - 1) pairs, the shape is [B, 2*dim*(num_modalities - 1), 64]
        fused_features = torch.cat(fused_pairs, dim=1)
        
        # Transpose to [B, 64, 2*dim*(num_modalities - 1)] for bottleneck projection
        fused_features = fused_features.transpose(1, 2)
        
        # Apply bottleneck projection to yield the final fused representation [B, 64, 64]
        fused_features = self.project_bottleneck(fused_features)
        
        return fused_features
