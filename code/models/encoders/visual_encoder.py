import torch
import torch.nn as nn
import torchvision

from models.adapter import vit_adapter_conv, vit_adapter_nlp
from models.visual_model import cnn_face

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
            intermediate_outputs = []
            for layer in self.ViT_layers:
                yield layer(faces)
            return
        
        for layer in self.ViT_layers:
            faces = layer(faces)
        return faces
