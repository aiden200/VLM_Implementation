import torch.nn as nn
import torch
from src.vision_transformer import VisionEncoderConfig

class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        self.config = config
        self.image_embed_dim = config.hidden_dim
        self.img_size = config.img_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.image_embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid" # no padding
        )

        # We are dividing the img by patch size, getting an embedding per 16x16 patch
        self.num_patches = (self.img_size // self.patch_size) ** 2 
        # Per patch, we need to generate an embedding
        self.position_encoder = nn.Embedding(self.num_patches, self.image_embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False
        )
    

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape

        # B, image_embed_dim, P_H, __W where P_H * P_W = num_patches 
        patch_embeddings = self.patch_embedding(pixel_values)

        # We want a 1D array, not a 2D shape, [B, E, 4, 4] -> [B, E, 16]
        patch_embeddings = patch_embeddings.flatten(2)

        # Add PE, [B, E, 16] -> [B, E, 16]
        patch_embeddings += self.position_encoder(self.position_ids)

        # NOTE: PE is not sinosoidal bc image patches are not sequential. 
        # We don't want the model to use the absolute positions, but learn how to use them

        return patch_embeddings


