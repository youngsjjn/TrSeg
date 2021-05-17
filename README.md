# TrSeg: Transformer for Semantic Segmentation

## Introduction

This repository is a PyTorch implementation of [TrSeg](https://hszhao.github.io/projects/psanet). This work is based on [semseg](https://github.com/hszhao/semseg/blob/1.0.0/README.md).

<img src="./figure/TrSeg_Architecture.png" width="900"/>

The codebase mainly uses ResNet50/101/152 as backbone and can be easily adapted to other basic classification structures. Sample experimented dataset is [Cityscapes](https://www.cityscapes-dataset.com).

## Requirement
Hardware: >= 44G GPU memory

Software: [PyTorch](https://pytorch.org/)>=1.0.0, python3

## Usage
For installation, follow installation steps below or recommend you to refer to the instructions described [here](https://github.com/hszhao/semseg/blob/1.0.0/README.md).

If you use multiple GPUs for training, [Apex](https://github.com/NVIDIA/apex) is required for synchronized training (such as Sync-BN).

For its pretrained model, you can download from [here](https://drive.google.com/file/d/1fxPpA_mkk1Ijur8HTnrkQtchVbYhzLyI/view?usp=sharing).

## Getting Started

### Installation

1. Clone this repository.
```
git clone https://github.com/youngsjjn/TrSeg.git
```

2. Install Python dependencies.
```
pip install -r requirements.txt
```

### Implementation
1. Download datasets (i.e. [Cityscapes](https://www.cityscapes-dataset.com)) and change the root of data path in [config](./config/cityscapes/cityscapes_transform101.yaml).
Download data list and pre-trained backbone models (ResNet50/101/152) [here](https://drive.google.com/open?id=15wx9vOM0euyizq-M1uINgN0_wjVRf9J3).

2. Train (Evaluation is included at the end of the training)
```
sh tool/train.sh cityscapes transform101
```

3. Test (Pre_trained
```
sh tool/test.sh cityscapes transform101
```

   |  Network (ResNet-101)  |     mIoU     |
   | :-------: | :----------: |
   | PSPNet  |    78.6    |
   | Deeplab-v3  |    79.3   |
   | TrSeg  |    [79.9](https://drive.google.com/file/d/1fxPpA_mkk1Ijur8HTnrkQtchVbYhzLyI/view?usp=sharing)    |
   
   
### Citation

You may want to cite:

```

```
