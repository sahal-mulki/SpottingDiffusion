# SpottingDiffusion <a target="_blank" href="https://colab.research.google.com/github/sahal-mulki/SpottingDiffusion/blob/main/SpottingDiffusion.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This repository is the official implementation of [SpottingDiffusion: Using transfer learning to detect Latent Diffusion Model-synthesized images](https://doi.org/10.59720/23-256)

![SpottingDiffusion tried on 3 random images from the dataset.](https://i.imgur.com/aSPB4nS.png)

# Table of Contents.
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Requirements](#requirements)
- [Training](#training)
- [Pretrained](#pretrained)
# Abstract

This study aims to present a novel method of detecting images made by “Latent Diffusion Models” as described by <a href="https://arxiv.org/abs/2112.10752"> Rombach et al.</a> 
The issue of differentiating AI generated images from real ones has recently become one of great importance and debate; as extremely realistic AI generated images are rapidly becoming easier to make and disseminate. 

The need of detecting these images arises when these technologies will inevitably be used to make misleading material with the intent of deceiving the human viewer. The authors of this study present a solution, an algorithmic way of differentiating images made by “Latent Diffusion Models” from real ones. In specific, we detail our research on detecting images produced by the “Stable Diffusion Latent Diffusion Model”. 

# Requirements

You may also easily use the Google Colab version of the trainer, which has training and downloading automatically built in.  <a target="_blank" href="https://colab.research.google.com/github/sahal-mulki/SpottingDiffusion/blob/main/SpottingDiffusion.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### To install dependencies:

`pip install -r requirements.txt`

### To download datasets for training and evaluation:

Login with kaggle API, and then,

```
kaggle datasets download sahalmulki/stable-diffusion-generated-images
kaggle datasets download sahalmulki/spottingdiffusion-testing-dataset

mkdir testing
unzip /content/spottingdiffusion-testing-dataset.zip -d /testing/
```

# Training

### Run this command after downloading the dataset for training the model as specified in the paper:
`python train.py 12 0.3 0.00001`

### Evaluate the model on the testing dataset using this command:
`python evaluate.py /full-path-to-dir/testing pretrained/pretrained-spotting-diffusion` 
Google Colab Notebook for Evaluation: <a target="_blank" href="https://colab.research.google.com/github/sahal-mulki/SpottingDiffusion/blob/main/SpottingDiffusion_Testing.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pretrained

### A pretrained model for SpottingDiffusion is available in the SavedModel format in the `pretrained` directory.

# Cite this:

```
@article{spottingdiffusionmulki2024,
title={Spottingdiffusion: Using transfer learning to detect latent diffusion model-synthesized images},
DOI={10.59720/23-256},
journal={Journal of Emerging Investigators},
author={Sahal Mulki, Muhammad and Adil Mulki, Sadaf},
year={2024},
month={Nov}
}
```
