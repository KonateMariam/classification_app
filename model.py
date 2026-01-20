import torch
import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    """
    Vision Transformer seul pour classification d’images
    """
    def __init__(self, num_classes=2):
        super().__init__()

        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,   # IMPORTANT pour le chargement
            num_classes=0
        )

        embed_dim = self.vit.num_features
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.vit(x)
        logits = self.classifier(features)
        return logits
