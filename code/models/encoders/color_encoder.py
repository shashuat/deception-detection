import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableSpectrogram(nn.Module):
    """
    Applies parallel temporal convolutions (with different kernel sizes)
    to learn frequency-domain representations from the raw RGB input.
    
    Input: (B, C, T, H, W)
    Output: (B, out_channels, T, H, W)
    """
    def __init__(self, in_channels=3, out_channels=3, kernel_sizes=[3, 5], stride=1, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, 
                      kernel_size=(k, 1, 1),
                      stride=(stride, 1, 1),
                      padding=(k // 2, 0, 0))
            for k in kernel_sizes
        ])
        # Fuse the parallel conv outputs
        self.bn = nn.BatchNorm3d(out_channels * len(kernel_sizes))
        self.relu = nn.ReLU()
        self.fuse = nn.Conv3d(out_channels * len(kernel_sizes), out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        # x: (B, C, T, H, W)
        outs = [conv(x) for conv in self.convs]  # Each: (B, out_channels, T, H, W)
        x_cat = torch.cat(outs, dim=1)            # (B, out_channels * num_kernels, T, H, W)
        x_cat = self.bn(x_cat)
        x_cat = self.relu(x_cat)
        x_cat = self.fuse(x_cat)                  # (B, out_channels, T, H, W)
        x_cat = self.dropout(x_cat)
        return x_cat

class MultiScaleConvProj(nn.Module):
    """
    Applies multiple parallel 3D convolutions (with different spatial kernel sizes)
    and then fuses them to extract multi-scale spatio-temporal token embeddings.
    
    Input: (B, C, T, H, W)
    Output: (B, embed_dim, T', H', W')
    """
    def __init__(self, in_channels, embed_dim, kernel_sizes=[(3, 32, 32), (3, 16, 16)], stride=(1, 32, 32), dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, embed_dim, kernel_size=k, stride=stride,
                      padding=(k[0]//2, k[1]//2, k[2]//2))
            for k in kernel_sizes
        ])
        self.proj = nn.Conv3d(embed_dim * len(kernel_sizes), embed_dim, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]  # Each: (B, embed_dim, T', H', W')
        x_cat = torch.cat(outs, dim=1)           # (B, embed_dim * num_scales, T', H', W')
        x_cat = self.proj(x_cat)
        x_cat = self.dropout(x_cat)
        return x_cat

import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorVariation3DViT(nn.Module):
    """
    3D ViT for color skin variation augmented with:
      1. Multi-scale feature extraction via a multi-scale conv projection.
      2. An enhanced learnable spectrogram module for frequency-domain features.
      3. Static (precomputed) positional encoding and extra transformer layers.
      
    Input: (B, T, C, H, W)
    Output: (B, 64, embed_dim) token sequence.
    """
    def __init__(self, embed_dim=512, num_heads=8, num_layers=2, num_context_layers=1,
                 dropout=0.3, input_shape=(16, 224, 224), desired_seq_len=64):
        """
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            num_context_layers: Additional transformer layers for contextual integration.
            dropout: Dropout rate.
            input_shape: Tuple (T, H, W) of expected input dimensions.
            desired_seq_len: The fixed output sequence length.
        """
        super().__init__()
        self.input_shape = input_shape  # (T, H, W)
        self.desired_seq_len = desired_seq_len

        # Enhanced spectrogram for temporal-frequency features.
        self.spectrogram = LearnableSpectrogram(
            in_channels=3, out_channels=3, kernel_sizes=[3, 5], stride=1, dropout=0.2
        )
        # Multi-scale convolutional projection to tokenize spatio-temporal features.
        self.conv_proj = MultiScaleConvProj(
            in_channels=3, embed_dim=embed_dim, 
            kernel_sizes=[(3, 32, 32), (3, 16, 16)],
            stride=(1, 32, 32), dropout=0.2
        )
        # Compute number of tokens from a dummy forward pass.
        self.num_tokens = self._compute_num_tokens()
        # Create fixed positional encoding.
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Additional contextual transformer layers.
        self.context_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_context_layers)
        # Final projection (applied tokenwise).
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _compute_num_tokens(self):
        """
        Use a dummy forward pass to compute the token count from the conv projection.
        """
        T, H, W = self.input_shape
        dummy = torch.zeros(1, 3, T, H, W)
        dummy = self.spectrogram(dummy)
        dummy = self.conv_proj(dummy)  # (1, embed_dim, T', H', W')
        _, _, T_prime, H_prime, W_prime = dummy.shape
        num_tokens = T_prime * H_prime * W_prime
        return num_tokens

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W).
        Returns:
            Tensor of shape (B, desired_seq_len, embed_dim).
        """
        B, T, C, H, W = x.shape
        # Permute input for 3D conv: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        # Apply learnable spectrogram.
        x = self.spectrogram(x)  # (B, 3, T, H, W)
        # Tokenize with multi-scale conv projection.
        x = self.conv_proj(x)    # (B, embed_dim, T', H', W')
        B, C_proj, T_prime, H_prime, W_prime = x.shape
        # Flatten spatial and temporal dims.
        num_tokens_current = T_prime * H_prime * W_prime
        x = x.flatten(2).transpose(1, 2)  # (B, num_tokens_current, embed_dim)
        
        # If current token count differs from the fixed one, interpolate positional embeddings.
        if num_tokens_current != self.num_tokens:
            pos_emb = F.interpolate(self.pos_embedding.transpose(1,2), 
                                    size=num_tokens_current, mode="linear", align_corners=False).transpose(1,2)
        else:
            pos_emb = self.pos_embedding
        # Add positional encoding.
        x = x + pos_emb
        
        # Process tokens with transformer layers.
        x = self.transformer(x)
        x = self.context_transformer(x)
        x = self.layer_norm(x) if hasattr(self, 'layer_norm') else x
        # Instead of global average pooling, interpolate tokens to fixed sequence length.
        if x.shape[1] != self.desired_seq_len:
            x = x.transpose(1,2)  # (B, embed_dim, num_tokens_current)
            x = F.interpolate(x, size=self.desired_seq_len, mode="linear", align_corners=False)
            x = x.transpose(1,2)  # (B, desired_seq_len, embed_dim)
        x = self.dropout(x)
        # Apply final projection tokenwise.
        return self.fc(x)

