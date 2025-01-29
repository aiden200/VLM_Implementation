import torch
import torch.nn as nn
from src.vision_transformer import VisionEncoderConfig


class Attention(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super.__init__()
        pass

class MLP(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super.__init__()
        
        self.fc1 = nn.Linear(config.hidden_dim, config.linear_dim)
        self.fc2 = nn.Linear(config.linear_dim, config.hidden_dim)
    
    def forward(self, patch_embeddings):
        # [B, 16, E]

        # [B, 16, E] -> [B, 16, linear_Dim]
        patch_embeddings = self.fc1(patch_embeddings)
        
        # gelu activation
        patch_embeddings = nn.functional.gelu(patch_embeddings, approximate="tanh")

        # [B, 16, linear_dim] -> [B, 16, E]
        out = self.fc2(patch_embeddings)

        return out


class VisionEncoder(nn.Model):
    def __init__(self, config: VisionEncoderConfig):
        super.__init__()
        self.config = config

        self.n_transformer_layers = self.config.num_transformer_layers
        self.n_attn_heads = self.config.num_attn_heads
        self.layer_norm_eps = self.config.layer_norm_eps
        self.attn_dropout = self.config.attn_dropout
        self.embed_dim = self.config.hidden_dim
        self.linear_dim = self.config.linear_dim

        self.layer_norm_1 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.attention = Attention(config)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.mlp = MLP(config)
    

    def forward(self, patch_embeddings):
        # [B, 16, E]

        resid = patch_embeddings

        # [B, 16, E] -> [B, 16, E]
        patch_embeddings = self.layer_norm_1(patch_embeddings)
        patch_embeddings = self.attention(patch_embeddings)

        patch_embeddings += resid
        resid = patch_embeddings

        patch_embeddings = self.layer_norm_2(patch_embeddings)
        patch_embeddings = self.mlp(patch_embeddings)

        out = resid + patch_embeddings

        # [B, 16, E]
        return out


