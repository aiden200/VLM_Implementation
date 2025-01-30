# VLM_Implementation

<!-- <div align="center">
    <img src="assets/Stable Diffusion.png">
</div> -->

<div align="center">

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-EE4C2C?style=flat-square&logo=pytorch"></a>

</div>

This project is an open-source implementation of a Vision-Language Model (VLM), designed to understand and generate meaningful tokens from image-text pairs. Our implementation integrates the main components of a VLM, including a transformer-based vision encoder, a text encoder for language understanding, a multimodal fusion mechanism, and a cross-attention framework to align visual and textual representations. Additionally, we incorporate contrastive learning and self-supervised objectives.

## Acknowledgments

Special thanks to Umar Jamil for his great video on this topic. We also extend our gratitude to the authors of the original VLM paper and the ViT paper for their work in the field of multimodal LLMs.

## Project Structure

The project is organized as follows:

```bash
── assets                           # Contains images for README
├── LICENSE
├── src
│   ├── vision_embedding.py         # Patch embedding models
│   ├── vision_encoder.py           # ViT Encoder
│   ├── vision_model.py             # Overall VLM model
│   └── vision_transformer.py       # Transformer blocks
├── pretrained                      # put pretrained model in this folder
├── README.md
├── requirements.txt
```

### 1. Environment Setup

We recommend creating a virtual environment and installing the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Training the Model

## 📅 Notes

<details>
  <summary><strong>Notes by Day</strong></summary>

- **Day 1**  
  ![Day 1 notes](assets/day_1_notes.jpg)

- **Day 2**  
  ![Day 2 notes](assets/day_2_notes.jpg)

- **Day 3**  
  ![Day 3 notes](assets/day_3_notes.jpg)

</details>

_Made with 💡 by Aiden Chang_
