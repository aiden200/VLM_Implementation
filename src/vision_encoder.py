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
        pass


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


