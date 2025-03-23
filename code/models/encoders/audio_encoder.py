# encoders/audio_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from models.adapter import w2v2_adapter_conv, w2v2_adapter_nlp


class WavEncoder(nn.Module):
    def __init__(self, num_encoders, adapter, adapter_type):
        super(WavEncoder, self).__init__()
        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        for p in model.parameters():
            p.requires_grad = False

        # Feature extractor and projection components
        self.feature_extractor = model.feature_extractor
        self.encoder_feature_projection = model.encoder.feature_projection
        self.encoder_pos_conv_embed = model.encoder.transformer.pos_conv_embed
        self.encoder_layer_norm = model.encoder.transformer.layer_norm
        self.encoder_dropout = model.encoder.transformer.dropout

        self.dropout = nn.Dropout(0.05)
        # Build Wav2Vec2 encoder layers
        audio_layer_list = []
        for i in range(num_encoders):
            if adapter:
                if adapter_type == 'nlp':
                    audio_layer_list.append(w2v2_adapter_nlp(transformer_encoder=model.encoder.transformer.layers[i]))
                else:
                    audio_layer_list.append(w2v2_adapter_conv(transformer_encoder=model.encoder.transformer.layers[i]))
            else:
                # Fine-tune encoder if not using adapters
                for p in model.encoder.transformer.layers[i].parameters():
                    p.requires_grad = True
                audio_layer_list.append(model.encoder.transformer.layers[i])

        self.transformer_layers = nn.ModuleList(audio_layer_list)

    def forward(self, audios, return_all_layers=False):
        """
        Process raw audio input through the feature extractor, projection,
        positional embedding, normalization, and then transformer layers.
        
        If return_all_layers is True, returns a generator
        for use in cross-modal fusion.
        """

        # Base audio feature extraction
        audios, _ = self.feature_extractor(audios, None)
        audios = self.encoder_feature_projection(audios)
        audios = self.encoder_pos_conv_embed(audios)
        audios = self.encoder_layer_norm(audios)
        audios = self.encoder_dropout(audios)

        if return_all_layers:
            return self._layer_generator(audios)
        
        for layer in self.transformer_layers:
            audios = layer(audios)
            audios = self.dropout(audios)
        return audios
    
    def _layer_generator(self, x):
        """Generator function to yield outputs of each transformer layer."""
        for layer in self.transformer_layers:
            x = layer(x)  # (batch, 64, embed)
            x = self.dropout(x)
            yield x

class WhisperTokenEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads, hidden_dim, vocab_size=51866, dropout=0.15):
        """
        Transformer-based encoder for Whisper tokens.

        :param embedding_dim: Dimensionality of token embeddings.
        :param num_layers: Number of transformer layers.
        :param num_heads: Number of attention heads.
        :param hidden_dim: Hidden dimension for MLP in transformer.
        :param vocab_size: Number of unique Whisper tokens (pre set at 51866 for whisper large V3)
        :param dropout: Dropout rate.
        """
        super(WhisperTokenEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # downsample from 256 to 64 (seq length)
        self.seq_projection = nn.Linear(256, 64)


        self.pos_embedding = nn.Parameter(torch.randn(1, 64, embedding_dim))  # (1, 64, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, return_all_layers=False):
        """
        Forward pass through Whisper Token Encoder.

        :param x: Tensor of Whisper tokens (batch_size, 256: seq_length)
        :param return_all_layers: If True, returns outputs from all transformer layers.
        :return: Encoded token representations.
        """

        # need to match the 64 frames samples
        x = self.token_embedding(x) # (batch, 256, embed)
        x = x.permute(0, 2, 1)
        x = x.float()

        x = self.seq_projection(x)  # (batch, embed, 64)
        x = x.permute(0, 2, 1)

        x += self.pos_embedding # (batch, seq, embed)

        x = self.layer_norm(x)
        if return_all_layers:
            return self._layer_generator(x)

        return self.transformer(x)  # (batch, 64, embed)
    
    def _layer_generator(self, x):
        """Generator function to yield outputs of each transformer layer."""
        for layer in self.transformer.layers:
            x = layer(x)  # (batch, 64, embed)
            yield x
