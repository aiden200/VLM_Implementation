import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from src.utils import process_images, add_image_tokens_to_prompt
from typing import List

IMAGENET_MEAN = [.5, .5, .5]
IMAGENET_STD = [.5, .5, .5]

class PaliGemmaProcessor:

    def __init__(self, tokenizer, img_seq_len, img_size):
        super().__init__()

        self.img_seq_len = img_seq_len
        self.img_size = img_size
        # Placeholder token used in our combined embedding
        self.IMAGE_TOKEN = "<Image>"

        additional_token = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(additional_token)
        
        # Location extraction tokens -> bounding boxes for object detection
        extra_tokens = [
            f"<Loc{i:04d}>" for i in range(1024)
        ]
        # segmentation tokens
        extra_tokens += [
            f"<seg{i:03d}>" for i in range(128)
        ]
        
        # Token info https://huggingface.co/blog/paligemma
        tokenizer.add_special_tokens(extra_tokens)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token, tokenizer.add_eos_token = False, False

        self.tokenizer = tokenizer
    
    # Given some prompts and images, tokenize them and create a valid input for PaliGemma
    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str="longest",
        truncation: bool = True
    ):
        # Process image into the size we care about, normalize, return it as a tensor
        pixel_values = process_images(
            images,
            size=(self.img_size, self.img_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            img_mean=IMAGENET_MEAN,
            img_std=IMAGENET_STD
        )

        # Stack to [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert to pytorch tensor
        pixel_values = torch.tensor(pixel_values)


        # Adding image token placeholders for the prompt
        input_text_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                img_seq_len=self.img_seq_len,
                img_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_text_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )

        out = {"pixel_values": pixel_values, **inputs}

        return out
