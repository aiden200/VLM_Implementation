import torch.nn as nn
import torch
from src.vision_transformer import VisionTransformer

class VisionEncoderConfig:
    def __init__(
        self,
        hidden_dim = 768,
        linear_dim = 3072,
        num_transformer_layers = 12,
        num_attn_heads = 12,
        num_channels = 3, #rgb
        img_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e-6,
        attn_dropout = 0.2,
        num_img_tokens = None, # how many image embeddings per image
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_dim = linear_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attn_heads = num_attn_heads
        self.num_channels = num_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attn_dropout = attn_dropout
        self.num_img_tokens = num_img_tokens

class VisionEncoderModel(nn.Model):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionTransformer(config)
    
    def forward(self, pixel_value):
        # [B, C, H, W] -> [B, patch_size, D]
        
        return self.vision_model(pixel_value)
        