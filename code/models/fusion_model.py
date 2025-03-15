import torch
import torch.nn as nn
from torch.nn import functional as F

from models.encoders.audio_encoder import WavEncoder
from models.encoders.visual_encoder import FaceEncoder


class CrossFusionModule(nn.Module):
    """
    Cross-modal fusion module: calculates cross-modal attention between audio and visual features
    and returns the fused feature representation.
    """
    def __init__(self, dim=256):
        super(CrossFusionModule, self).__init__()

        # Project features to common dimension
        self.project_audio = nn.Linear(768, dim)
        self.project_vision = nn.Linear(768, dim)
        
        # Learnable correlation weights
        self.corr_weights = nn.Parameter(torch.empty(dim, dim).uniform_(-0.1, 0.1))
        
        # Project concatenated features to bottleneck
        self.project_bottleneck = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.LayerNorm(64, eps=1e-05, elementwise_affine=True),
            nn.ReLU()
        )

    def forward(self, audio_feat, visual_feat):
        """
        Args:
            audio_feat: Audio features [batch_size, 64, 768]
            visual_feat: Visual features [batch_size, 64, 768]
        
        Returns:
            Fused features [batch_size, 64, 64]
        """
        # Ensure inputs are tensors
        if isinstance(audio_feat, tuple):
            audio_feat = audio_feat[0]
        if isinstance(visual_feat, tuple):
            visual_feat = visual_feat[0]
            
        # Project features to common dimension
        audio_feat = self.project_audio(audio_feat)  # [batch_size, 64, 256]
        visual_feat = self.project_vision(visual_feat)  # [batch_size, 64, 256]

        # Prepare for matrix multiplication
        visual_feat_t = visual_feat.transpose(1, 2)  # [batch_size, 256, 64]

        # Calculate correlation matrix
        a1 = torch.matmul(audio_feat, self.corr_weights)  # [batch_size, 64, 256]
        cc_mat = torch.bmm(a1, visual_feat_t)  # [batch_size, 64, 64]

        # Calculate attention weights
        audio_att = F.softmax(cc_mat, dim=1)  # [batch_size, 64, 64]
        visual_att = F.softmax(cc_mat.transpose(1, 2), dim=1)  # [batch_size, 64, 64]
        
        # Apply attention
        atten_audiofeatures = torch.bmm(audio_feat.transpose(1, 2), audio_att)  # [batch_size, 256, 64]
        atten_visualfeatures = torch.bmm(visual_feat_t, visual_att)  # [batch_size, 256, 64]
        
        # Skip connections
        atten_audiofeatures = atten_audiofeatures + audio_feat.transpose(1, 2)  # [batch_size, 256, 64]
        atten_visualfeatures = atten_visualfeatures + visual_feat_t  # [batch_size, 256, 64]

        # Concatenate and project to bottleneck
        fused_features = torch.cat((atten_audiofeatures, atten_visualfeatures), dim=1)  # [batch_size, 512, 64]
        fused_features = fused_features.transpose(1, 2)  # [batch_size, 64, 512]
        fused_features = self.project_bottleneck(fused_features)  # [batch_size, 64, 64]

        return fused_features


class Fusion(nn.Module):
    """
    Multimodal fusion model that combines audio and visual features
    with different fusion strategies and adapter options.
    """
    def __init__(self, fusion_type, num_encoders, adapter, adapter_type, multi=False):
        super(Fusion, self).__init__()

        assert fusion_type in ["concat", "cross2"], "The fusion type should be 'concat' or 'cross2' "
        
        self.fusion_type = fusion_type  # "concat" or "cross2"
        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type
        self.multi = multi  # multitask learning with multiple losses

        self.wav_encoder = WavEncoder(num_encoders, adapter, adapter_type)
        self.face_encoder = FaceEncoder(num_encoders, adapter, adapter_type)

        # Fusion and classification components
        if self.fusion_type == "concat":
            # Simple concatenation of features
            self.classifier = nn.Sequential(nn.Linear(768 * 2, 2))

        elif self.fusion_type == "cross2":
            # Cross-modal fusion
            cross_conv_layer = []
            for i in range(self.num_encoders):
                cross_conv_layer.append(CrossFusionModule(dim=256))
            self.cross_conv_layer = nn.ModuleList(cross_conv_layer)
            
            # Main classifier
            self.classifier = nn.Sequential(
                nn.Linear(64 * self.num_encoders, 64),
                nn.Dropout(p=0.5),
                nn.Linear(64, 2)
            )
            
            # Optional multitask learning classifiers
            if self.multi:
                self.audio_classifier = nn.Linear(768, 2)
                self.vision_classifier = nn.Linear(768, 2)

    def forward(self, audios, faces, whisper_tokens, bert_embedding):
        
        if self.fusion_type == "concat":
            audio_features = self.wav_encoder(audios, return_all_layers=False)
            face_features  = self.face_encoder(faces, return_all_layers=False)

            fused_output = torch.cat((audio_features, face_features), dim=-1)

        elif self.fusion_type == "cross2":
            audio_layers = self.wav_encoder(audios, return_all_layers=True) # generators
            face_layers  = self.face_encoder(faces, return_all_layers=True)
            
            # assert len(audio_layers) == len(face_layers) == len(self.cross_conv_layer), "Unmatched number of encoder layers for audio, vision, and cross fusion"
            
            # cross fusion
            feat_ls = []
            for audio_layer, face_layer, cross_conv in zip(audio_layers, face_layers, self.cross_conv_layer):
                fused_features = cross_conv(audio_layer, face_layer)
                feat_ls.append(fused_features)

            # concatenate features from all layers
            fused_output = torch.cat(feat_ls, dim=-1)
        else:
            raise ValueError(f"Undefined fusion type: {self.fusion_type}")

        # classification
        logits = self.classifier(fused_output)

        # multitask learning
        if self.multi:
            a_logits = self.audio_classifier(audios)
            v_logits = self.vision_classifier(faces)
            return torch.mean(logits, 1), torch.mean(a_logits, 1), torch.mean(v_logits, 1)
        else:
            return torch.mean(logits, 1), None, None