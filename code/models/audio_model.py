import torch
import torch.nn as nn
import torchaudio
from models.adapter import w2v2_adapter_nlp, w2v2_adapter_conv


class W2V2_Model(nn.Module):
    def __init__(self, num_encoders, adapter, adapter_type):
        super(W2V2_Model, self).__init__()

        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type

        # Load pretrained wav2vec2 model
        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        # Freeze base model weights
        for p in model.parameters(): 
            p.requires_grad = False

        # Pretrained CNN feature extractor
        self.FEATURE_EXTRACTOR = model.feature_extractor

        # Pretrained feature projection + pos encoding
        self.FEATURE_PROJECTOR = nn.Sequential(
            model.encoder.feature_projection,
            model.encoder.transformer.pos_conv_embed,
            model.encoder.transformer.layer_norm,
            model.encoder.transformer.dropout,
        )

        # Build w2v2 encoder with desired number of encoder layers
        layer_list = []

        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == 'nlp':
                    layer_list.append(w2v2_adapter_nlp(transformer_encoder=model.encoder.transformer.layers[i]))
                else:
                    layer_list.append(w2v2_adapter_conv(transformer_encoder=model.encoder.transformer.layers[i]))
            else:
                # Fine-tune encoder layers if not using adapters
                for p in model.encoder.transformer.layers[i].parameters(): 
                    p.requires_grad = True
                layer_list.append(model.encoder.transformer.layers[i])

        self.TRANSFORMER = nn.Sequential(*layer_list)

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 2)
        )

    def forward(self, x):
        # Extract features using CNN feature extractor
        features, _ = self.FEATURE_EXTRACTOR(x, None)
        
        # Project features and add positional encoding
        projections = self.FEATURE_PROJECTOR(features)
        
        # Process through transformer encoders
        output_tokens = self.TRANSFORMER(projections)
        
        # Final classification
        logits = self.classifier(output_tokens)
        
        # Average predictions over all tokens
        return torch.mean(logits, 1)