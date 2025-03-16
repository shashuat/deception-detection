import torch
import torch.nn as nn

from models.encoders.audio_encoder import WavEncoder


class W2V2_Model(nn.Module):
    def __init__(self, num_encoders, adapter, adapter_type):
        super(W2V2_Model, self).__init__()

        self.wav_encoder = WavEncoder(num_encoders, adapter, adapter_type)
        self.classifier = nn.Sequential(
            nn.Linear(768, 2)
        )

    def forward(self, x):        
        output_tokens = self.wav_encoder(x)
        logits = self.classifier(output_tokens)
        
        # Average predictions over all tokens
        return torch.mean(logits, 1) # strange ??

