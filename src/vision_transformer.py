import torch.nn as nn
import torch
from src.vision_model import VisionEncoderConfig
from src.vision_embedding import VisionEmbeddings
from src.vision_encoder import VisionEncoder


class VisionTransformer(nn.Model):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_dim
        self.embeddings = VisionEmbeddings(self.config)
        self.encoder = VisionEncoder(self.config)
        self.post_ln = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, x):
        
        embeddings = self.embeddings(x)
        
        post_transformer_hidden_state = self.encoder(embeddings)
        out = self.post_ln(post_transformer_hidden_state)
        
        return out