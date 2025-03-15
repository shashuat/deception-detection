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


class ViT_model(nn.Module):
    """Vision Transformer model for processing face frames"""
    def __init__(self, num_encoders, adapter, adapter_type):
        super(ViT_model, self).__init__()

        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type

        # Project CNN features to ViT dimension
        self.projection = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU(),
        )

        # Load ImageNet pretrained ViT Base 16 and freeze all parameters
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        vit_b_16 = torchvision.models.vit_b_16(weights=weights)
        for p in vit_b_16.parameters(): 
            p.requires_grad = False
            
        # Extract encoder only
        vit = vit_b_16.encoder
        
        # Add learnable positional embedding for 64 tokens
        self.pos_embedding = nn.Parameter(torch.empty(1, 64, 768).normal_(std=0.02))

        # Build ViT encoder layers
        layer_list = []

        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == 'nlp':
                    layer_list.append(vit_adapter_nlp(transformer_encoder=vit.layers[i]))
                else:
                    layer_list.append(vit_adapter_conv(transformer_encoder=vit.layers[i]))
            else:
                # Fine-tune encoder layers if not using adapters
                for p in vit.layers[i].parameters(): 
                    p.requires_grad = True
                layer_list.append(vit.layers[i])

        # Add final layer normalization
        layer_list.append(nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True))

        # Assign models for forward pass
        self.cnn_feature_extractor = cnn_face()
        self.ViT_Encoder = nn.Sequential(*layer_list)

        # Classification head for deception detection
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
        )

    def forward(self, x):
        # Process all frames with CNN
        b_s, no_of_frames, C, H, W = x.shape
        x = torch.reshape(x, (b_s * no_of_frames, C, H, W))
        features = self.cnn_feature_extractor(x)
        features = torch.reshape(features, (b_s, no_of_frames, 256))

        # Project features and add positional embeddings
        projections = self.projection(features) + self.pos_embedding
        
        # Process through ViT encoder
        vit_features = self.ViT_Encoder(projections)
        
        # Final classification
        logits = self.classifier(vit_features)

        # Average predictions over all tokens
        return torch.mean(logits, 1)