import torch
import torch.nn as nn
from src.gemma import PaliGemmaConfig


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.lin = nn.Linear(config.vision_config.hidden_dim, config.projection_dim, bias=True)


    # Project Image hidden layer into the language dimension 
    def forward(self, img_features):
        # [B, Num Patches, E] -> [B, Num Patches, Proj Dim]
        out = self.lin(img_features)
        return out