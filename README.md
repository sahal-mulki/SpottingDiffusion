# SpottingDiffusion <a target="_blank" href="https://colab.research.google.com/github/sahal-mulki/SpottingDiffusion/blob/main/SpottingDiffusion.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This repository is the official implementation of SpottingDiffusion : A CNN-based method of detecting AI generated images.

![SpottingDiffusion tried on 3 random images from the dataset.](https://i.imgur.com/aSPB4nS.png)

# Table of Contents.
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Requirements](#requirements)
- [Training](#training)
- 
# Abstract

This study aims to present a novel method of detecting images made by “Latent Diffusion Models” as described by <a href="https://arxiv.org/abs/2112.10752"> Rombach et al.</a> 
The issue of differentiating AI generated images from real ones has recently become one of great importance and debate; as extremely realistic AI generated images are rapidly becoming easier to make and disseminate. 

The need of detecting these images arises when these technologies will inevitably be used to make misleading material with the intent of deceiving the human viewer. The authors of this study present a solution, an algorithmic way of differentiating images made by “Latent Diffusion Models” from real ones. In specific, we detail our research on detecting images produced by the “Stable Diffusion Latent Diffusion Model”. 

# Requirements

PS: You can also easily use the Google Colab version of the trainer, which has evaluation and downloading automatically built in. 

<a target="_blank" href="https://colab.research.google.com/github/sahal-mulki/SpottingDiffusion/blob/main/SpottingDiffusion.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
`python evaluate.py /testing SpottingDiffusion`
