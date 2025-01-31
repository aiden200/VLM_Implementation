import torch
import torch.nn as nn
from src.vision_transformer import VisionEncoderConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super.__init__()

        self.config = config
        self.embed_dim = config.hidden_dim
        self.num_heads = config.num_attn_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.dropout = config.attn_dropout
        self.q_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_w = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)
        pass

    def forward(self, x):
        # [B, 16, E]
        B, P, E = x.shape

        # [B, 16, E]
        q, k, v = self.q_w(x), self.k_w(x), self.v_w(x)

        # [B, Num_head, 16, head_dim], where Num_Head * head_dim = E
        q = q.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)

        # Switch head_dim and patch, so we get 16x16 attention weights qk/(d^2)
        qk = torch.matmul(q, k.transpose(2,3)) * (self.head_dim**-.5)

        qk = nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)
        qk = nn.functional.dropout(qk, p=self.dropout, training=self.training)

        # [B, H, 16, H_D]
        attention_values = torch.matmul(qk, v)

        # [B, 16, E]
        out = attention_values.transpose(1, 2).view(B, P, E)

        # [B, 16, E]
        out = self.out_projection(out)

        return out


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
        super().__init__()
        self.config = config

        self.n_transformer_layers = self.config.num_transformer_layers
        self.n_attn_heads = self.config.num_attn_heads
        self.layer_norm_eps = self.config.layer_norm_eps
        self.attn_dropout = self.config.attn_dropout
        self.embed_dim = self.config.hidden_dim
        self.linear_dim = self.config.linear_dim

        self.layer_norm_1 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
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


