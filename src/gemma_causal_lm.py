import torch
import torch.nn as nn
from src.gemma import GemmaConfig

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.model_layers = nn.ModuleList(
            [GemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self,
        attention_mask,
        position_ids,
        input_embeds,
        kv_cache
    ):
        hidden_states = input_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for layer in self.model_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache
            )
        
        out = self.norm(hidden_states)

        return out



class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        attention_mask = None,
        position_ids = None,
        input_embeds = None,
        kv_cache = None
        ):
        # input_embeds = [B, S, H] -> outputs: [B, S, H]

        # Process the image and text embeddings
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache
        ) 
        hidden_states = outputs

        # Now, predict the next vocab
        logits = self.lm_head(hidden_states)
        logits = logits.float()