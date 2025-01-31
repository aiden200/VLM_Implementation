import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union


def resize(img, size, resample=None, reducing_gap=None):
    h, w = size
    return img.resize((w, h), resample=resample, reducing_gap=reducing_gap)

def rescale(img, scale, dtype:np.dtype=np.float32):
    rescaled_img = img*scale
    return rescaled_img.astype(dtype)

def normalize(img, mean, std):
    mean = np.array(mean, dtype=img.dtype)
    std = np.array(std, dtype=img.dtype)

    return (img - mean) / std


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    img_seq_len,
    img_token
):
    # Image tokens * how many sequence tokens, beggining of sentence, prompt of the user (prefix_prompt)
    return f"{img_token*img_seq_len}{bos_token}{prefix_prompt}\n"


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    img_mean: Optional[Union[float, List[float]]] = None,
    img_std: Optional[Union[float, List[float]]] = None
):
    h, w = size[0], size[1]
    imgs = [resize(image=image, size=(h,w), resample=resample) for image in images]
    imgs = [np.array(img) for img in imgs]
    
    # Rescale pixel value to be in [0, 1]
    imgs = [rescale(img, scale=rescale_factor) for img in imgs]
    # Normalize, mean 0 and std 1
    imgs = [normalize(img, mean=img_mean, std=img_std) for img in imgs]

    # [H, W, C] -> [C, H, W]
    imgs = [img.transpose(2, 0, 1) for img in imgs]

    return imgs