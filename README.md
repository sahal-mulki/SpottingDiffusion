# SpottingDiffusion

This repository is the official implementation of SpottingDiffusion : A CNN-based method of detecting AI generated images.

![SpottingDiffusion tried on 3 random images from the dataset.](https://i.imgur.com/aSPB4nS.png)

# Requirements

### To install dependencies:

`pip install -r requirements.txt`

### To download datasets for training and evaluation:

Login with kaggle API, and then,

```
kaggle datasets download sahalmulki/stable-diffusion-generated-images/versions/3
kaggle datasets download sahalmulki/spottingdiffusion-testing-dataset

mkdir testing
unzip /content/spottingdiffusion-testing-dataset.zip -d /testing/
```

# Training

### Run this command after downloading the dataset for training the model as specified in the paper:
`python train.py 12 0.3 0.00001`

### Evaluate the model on the testing dataset using this command:
`python evaluate.py /testing SpottingDiffusion`
