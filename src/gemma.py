import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from src.vision_model import VisionEncoderConfig, VisionEncoderModel

class PaliGemma(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = VisionEncoderModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMMProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLm(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    