# encoders/visual_encoder.py

import torch
import torch.nn as nn
import torchvision

from models.adapter import vit_adapter_conv, vit_adapter_nlp


class conv2d_block(nn.Module):
    """Simple CNN block with batch normalization and residual connections"""
    def __init__(self, in_channels, out_channels, pad='same', k=3, s=1):
        super(conv2d_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=pad, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class cnn_face(nn.Module):
    """CNN for extracting facial features"""
    def __init__(self):
        super(cnn_face, self).__init__()

        # Initial conv layer
        self.conv1 = conv2d_block(3, 64, k=7, pad=3, s=2)
        self.layer1 = nn.Sequential(
            conv2d_block(64, 64),
            conv2d_block(64, 64),
        )

        # Second stage
        self.conv2 = conv2d_block(64, 128, k=3, pad=1, s=2)
        self.layer2 = nn.Sequential(
            conv2d_block(128, 128),
            conv2d_block(128, 128),
        )

        # Third stage
        self.conv3 = conv2d_block(128, 256, k=3, pad=1, s=2)
        self.layer3 = nn.Sequential(
            conv2d_block(256, 256),
            conv2d_block(256, 256),
        )

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # First block with residual
        x = self.conv1(x)
        x = self.layer1(x) + x
        
        # Second block with residual
        x = self.conv2(x)
        x = self.layer2(x) + x
        
        # Third block with residual
        x = self.conv3(x)
        x = self.layer3(x) + x
        
        # Return pooled features
        return self.avg_pool(x)


class FaceEncoder(nn.Module):
    def __init__(self, num_encoders, adapter, adapter_type):
        super(FaceEncoder, self).__init__()
        # Projection layer for CNN features
        self.projection = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU(),
        )

        # Load pretrained ViT and freeze its parameters
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        vit_b_16 = torchvision.models.vit_b_16(weights=weights)
        for p in vit_b_16.parameters():
            p.requires_grad = False

        vit = vit_b_16.encoder
        self.pos_embedding = nn.Parameter(torch.empty(1, 64, 768).normal_(std=0.02))
        self.cnn_feature_extractor = cnn_face()

        # Build ViT encoder layers
        face_layer_list = []
        for i in range(num_encoders):
            if adapter:
                if adapter_type == 'nlp':
                    face_layer_list.append(vit_adapter_nlp(transformer_encoder=vit.layers[i]))
                else:
                    face_layer_list.append(vit_adapter_conv(transformer_encoder=vit.layers[i]))
            else:
                # Fine-tune encoder if not using adapters
                for p in vit.layers[i].parameters():
                    p.requires_grad = True
                face_layer_list.append(vit.layers[i])
        self.ViT_layers = nn.ModuleList(face_layer_list)        

    def forward(self, faces, return_all_layers=False):
        """
        Process visual frames through the face encoder.
        :param faces: Tensor of shape (batch, num_frames, C, H, W).
        :return: Processed face features.
        """
        b_s, no_of_frames, C, H, W = faces.shape

        # reshape to combine batch and frame dimensions for CNN processing
        faces = faces.view(b_s * no_of_frames, C, H, W)
        
        faces = self.cnn_feature_extractor(faces) # (b_s * no_of_frames, 256)
        faces = faces.view(b_s, no_of_frames, 256)
        
        faces = self.projection(faces) + self.pos_embedding
        
        if return_all_layers:
            return self._layer_generator(faces)
        
        for layer in self.ViT_layers:
            faces = layer(faces)
        return faces
    
    def _layer_generator(self, x):
        """Generator function to yield outputs of each transformer layer."""
        for layer in self.ViT_layers:
            x = layer(x)  # (batch, 64, embed)
            yield x
