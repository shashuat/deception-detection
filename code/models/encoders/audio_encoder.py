
import torch
import torch.nn as nn
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
        
        If return_all_layers is True, returns a tuple (final_output, list_of_intermediate_outputs)
        for use in cross-modal fusion.
        """

        # Base audio feature extraction
        audios, _ = self.feature_extractor(audios, None)
        audios = self.encoder_feature_projection(audios)
        audios = self.encoder_pos_conv_embed(audios)
        audios = self.encoder_layer_norm(audios)
        audios = self.encoder_dropout(audios)

        if return_all_layers:
            for layer in self.transformer_layers:
                yield layer(audios)
            return
        
        for layer in self.transformer_layers:
            audios = layer(audios)
        return audios
    