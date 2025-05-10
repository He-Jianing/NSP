# NSP

This is the code for the paper “Improving Prediction Certainty Estimation for Reliable Early Exiting via Null Space Projection“.

## Installation

This repository has been tested on Python 3.6.13, PyTorch 1.8.0, and Cuda 11.1. It is recommended to use a conda environment, for example:

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch
```

After installing the required environment, clone this repository and install the following requirements:

```
git clone https://github.com/He-Jianing/NSP.git
cd code_ijcai25_nsp
pip install -r ./requirements.txt
```

## Usage

There are two scripts in the `scripts` folder, which can be run from the repository root, e.g., `bash scripts/train.sh`.

#### train.sh

This is for fine-tuning NSP models.

#### eval_CAP.sh

This is for evaluating fine-tuned NSP models with various early exiting thresholds.
