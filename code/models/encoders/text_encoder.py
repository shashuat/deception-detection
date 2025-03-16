import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=4, transformer_heads=6, transformer_hidden_dim=1024, dropout=0.1):
        super(TextEncoder, self).__init__()
        
        # Downsample the sequence using a 1D convolution from 256 to 64 tokens.
        self.projection = nn.Linear(256, 64)
        
        # Learnable positional embeddings for the downsampled sequence.
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, embedding_dim))
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Transformer encoder layers to further process the downsampled tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x, return_all_layers=False):
        # x: (batch, embed, seq_length:256)
        x = self.projection(x)        # (b, embed, 64)
        x = x.transpose(1, 2)  # (batch, 64, embed)
        
        x = self.layer_norm(x + self.pos_embedding)
        if return_all_layers:
            return self._layer_generator(x)
        
        for layer in self.transformer.layers:
            x = layer(x)
        return x # (batch, 64, embed)
    
    def _layer_generator(self, x):
        """Generator function to yield outputs of each transformer layer."""
        for layer in self.transformer.layers:
            x = layer(x)  # (batch, 64, embed)
            yield x
