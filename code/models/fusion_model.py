import torch
import torch.nn as nn
import torchaudio
import torchvision
from models.adapter import w2v2_adapter_nlp, w2v2_adapter_conv, vit_adapter_nlp, vit_adapter_conv
from models.visual_model import cnn_face
from torch.nn import functional as F


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
        self.fusion_type = fusion_type  # "concat" or "cross2"
        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type
        self.multi = multi  # multitask learning with multiple losses

        # Audio model components
        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        for p in model.parameters():
            p.requires_grad = False

        # Feature extractor
        self.FEATURE_EXTRACTOR = model.feature_extractor

        # Feature projection + positional encoding
        self.encoder_feature_projection = model.encoder.feature_projection
        self.encoder_pos_conv_embed = model.encoder.transformer.pos_conv_embed
        self.encoder_layer_norm = model.encoder.transformer.layer_norm
        self.encoder_dropout = model.encoder.transformer.dropout

        # Build Wav2Vec2 encoder layers
        audio_layer_list = []
        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == 'nlp':
                    audio_layer_list.append(w2v2_adapter_nlp(transformer_encoder=model.encoder.transformer.layers[i]))
                else:
                    audio_layer_list.append(w2v2_adapter_conv(transformer_encoder=model.encoder.transformer.layers[i]))
            else:
                # Fine-tune encoder if not using adapters
                for p in model.encoder.transformer.layers[i].parameters(): 
                    p.requires_grad = True
                audio_layer_list.append(model.encoder.transformer.layers[i])

        self.TRANSFORMER = nn.ModuleList(audio_layer_list)

        # Visual model components
        self.projection = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU(),
        )

        # Load pretrained ViT
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        vit_b_16 = torchvision.models.vit_b_16(weights=weights)
        for p in vit_b_16.parameters():
            p.requires_grad = False

        # Extract encoder only
        vit = vit_b_16.encoder

        # Learnable positional embedding for 64 tokens
        self.pos_embedding = nn.Parameter(torch.empty(1, 64, 768).normal_(std=0.02))

        # Build ViT encoder layers
        face_layer_list = []
        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == 'nlp':
                    face_layer_list.append(vit_adapter_nlp(transformer_encoder=vit.layers[i]))
                else:
                    face_layer_list.append(vit_adapter_conv(transformer_encoder=vit.layers[i]))
            else:
                # Fine-tune encoder if not using adapters
                for p in vit.layers[i].parameters(): 
                    p.requires_grad = True
                face_layer_list.append(vit.layers[i])

        # CNN feature extractor and ViT encoder
        self.cnn_feature_extractor = cnn_face()
        self.ViT_Encoder = nn.ModuleList(face_layer_list)

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
        else:
            # Default to concatenation
            self.classifier = nn.Sequential(nn.Linear(768 * 2, 2))

    def forward(self, x, y):
        # Process audio
        x, _ = self.FEATURE_EXTRACTOR(x, None)
        
        # Apply feature projection
        x = self.encoder_feature_projection(x)
        # Apply positional embedding
        x = self.encoder_pos_conv_embed(x)
        # Apply layer norm and dropout
        x = self.encoder_layer_norm(x)
        audios = self.encoder_dropout(x)

        # Process visual frames
        b_s, no_of_frames, C, H, W = y.shape
        y = torch.reshape(y, (b_s * no_of_frames, C, H, W))
        faces = self.cnn_feature_extractor(y)
        faces = torch.reshape(faces, (b_s, no_of_frames, 256))
        
        # Apply projection and positional embedding
        faces = self.projection(faces) + self.pos_embedding

        # Apply different fusion strategies
        feat_ls = []
        if self.fusion_type == "concat":
            # Process through encoders
            for audio_net in self.TRANSFORMER:
                audios = audio_net(audios)
                
            for visual_net in self.ViT_Encoder:
                faces = visual_net(faces)
                
            # Simple concatenation
            fused_output = torch.cat((audios, faces), dim=-1)
        elif self.fusion_type == "cross2":
            # Cross-modal fusion at each encoder layer
            assert len(self.TRANSFORMER) == len(self.ViT_Encoder), "Unmatched encoders between audio and face"
            
            for audio_net, visual_net, cross_conv in zip(self.TRANSFORMER, self.ViT_Encoder, self.cross_conv_layer):
                audios = audio_net(audios)
                faces = visual_net(faces)
                fused_features = cross_conv(audios, faces)
                feat_ls.append(fused_features)
                
            # Concatenate features from all layers
            fused_output = torch.cat(feat_ls, dim=-1)
        else:
            raise ValueError(f"Undefined fusion type: {self.fusion_type}")

        # Main classification
        logits = self.classifier(fused_output)

        # Handle multitask learning
        if self.multi:
            a_logits = self.audio_classifier(audios)
            v_logits = self.vision_classifier(faces)
            return torch.mean(logits, 1), torch.mean(a_logits, 1), torch.mean(v_logits, 1)
        else:
            return torch.mean(logits, 1), None, None