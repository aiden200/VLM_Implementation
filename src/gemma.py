import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from src.vision_model import VisionEncoderConfig, VisionEncoderModel
from src.gemma_mm_projector import PaliGemmaMultiModalProjector

class GemmaConfig():
    # From HF PaliGemma model, config.json file
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers, # num of transformer layers
        num_attn_layers, # number of query heads (Grouped query attention)
        num_key_value_heads, # number of key & value heads (Grouped query attention)
        head_dim=256,
        max_positional_embedding=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attn_bias=False,
        attn_dropout=0.0,
        pad_token_id=None
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_layers = num_attn_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_positional_embedding = max_positional_embedding
        self.rms_norm_eps = rms_norm_eps,
        self.rope_theta = rope_theta,
        self.attn_bias = attn_bias,
        self.attn_dropout = attn_dropout,
        self.pad_token_id = pad_token_id
        
        
        


class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        img_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        # kwargs**
        ):
        super().__init__()
        
        self.vision_config = VisionEncoderConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_img_tokens = (self.vision_config.img_size // self.vision_config.patch_size) ** 2
        
        self.ignore_index = ignore_index
        
        # <Image> token
        self.img_token_index = img_token_index
        # The linear projection dim of img tokens to embedding dim
        self.projection_dim = projection_dim
        # Language model embedding size
        self.hidden_size = hidden_size,
        self.pad_token_id = pad_token_id
        
        


class PaliGemma(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = VisionEncoderModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLm(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weight(self):
        return self.language_model.tie_weights()
    
    def merge_txt_and_img_tokens(self, img_features, inputs_embedded, input_ids, attn_mask, kv_cache=None):
        I_B, I_P, embed_dim = img_features.shape
        T_B, sequence_length = input_ids.shape # the positions in the embedding
        dtype, device = inputs_embedded.dtype, inputs_embedded.device
        '''
        Based on the image token placeholders in inputs_embedded, we want to inject the image features in the correct place.
        Then, we create an attention mask (0 indicates we are NOT masking out, we add -inf if we are masking)
        Then we create 
        ''' 
        
        # scaling [B, S, H] 
        scaled_img_features = img_features / (self.config.hidden_size**.5)
        
        # Combine the tokens of the image and text
        final_input_embedding = torch.zeros(T_B, sequence_length, embed_dim, dtype=dtype, device=device)
        
        # Anything not an image token or a pad token is good to go
        text_mask = (input_ids != self.config.img_token_index) & (input_ids != self.pad_token_id)
        
        # Image mask
        img_mask = (input_ids == self.config.img_token_index)
        
        # pad mask
        pad_mask = (input_ids == self.config.pad_token_id)
        
        # Expand these masks to the dimension we specified (embed_dim)
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        img_mask = img_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        
        # Where the text mask is 1, copy the text token into it 
        final_input_embedding = torch.where(text_mask, inputs_embedded, final_input_embedding)
        # Can't use torch.where since scaled_img_featuers have different dimension
        final_input_embedding = final_input_embedding.masked_scatter(img_mask, scaled_img_features)
        final_input_embedding = torch.where(pad_mask, torch.zeros_like(final_input_embedding), final_input_embedding)
        

        # During inference, the KV Cache only needs to access the final token, which is conditioned on all previous tokens, so we don't need to mask
        # During training, we do need to mask out the true tokens
        if kv_cache is None or kv_cache.num_items() == 0:
            # No mask since our kv cache has nothing in it, create a tensor w/ mask
            # PaliGemma's Text and Prefix tokens will ALWAYS be accessible
            causal_mask = torch.full((T_B, sequence_length, sequence_length), fill_value=0, dtype=dtype, device=device)
        else:
            # We have the number of kv_cache items and we are adding in the query
            kv_cache_length = kv_cache.num_items() + sequence_length
            
            causal_mask = torch.full((T_B, sequence_length, kv_cache_length), fill_value=0, dtype=dtype, device=device)
        
        # Add the number of attention heads
        # [B, Q_Len, KV_Len] -> [B, Attn_Head, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # Applying rotational position encoding
        if kv_cache is not None and kv_cache.num_times() > 0:
            # Saving the position in integers of each token -> [0, 1, 2, ..., 255, 256] - sequence length
            position_ids = attn_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create position_ids based on the size of the attention mask
            position_ids = (attn_mask.cumsum(-1)).masked_fill((attn_mask == 0), 1).to(device)
        
        return final_input_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple:
        # No padding
        assert torch.all(attention_mask == 1), "Padding has not yet been implemented"
        
        # D is the textual embedding size, E is the image embedding size
        
        # Embed the prompt
        # [B, Seq_len, D]
        inputs_embedded = self.language_model.get_input_embeddings()(input_ids)
        
        # Embed the image tokens
        # [B, C, H, W] -> [B, 16, E]
        img_features = self.vision_tower(pixel_values.to(inputs_embedded.dtype))
        
        # Project the image tokens to the textual embedding size, D
        # [B, 16, E] -> [B, 16, D]
        img_features = self.multi_modal_projector(img_features)
        
        # Combine the text and visual tokens
        inputs_embedded, attention_mask, position_ids = self.merge_txt_and_img_tokens(img_features, inputs_embedded, kv_cache)

        # Now our input embeddings are multimodal and projected into the language model dimension
        # Feed it through the "multimodal" transformer model to generate outputs
        out = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embedded=inputs_embedded,
            kv_cache=kv_cache
        )
        
        return out