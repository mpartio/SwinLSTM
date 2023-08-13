# SwinLSTM: A new recurrent cell for spatiotemporal modeling

This repository contains the official PyTorch implementation of the following paper:

SwinLSTM: Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM  **(ICCV 2023 Oral)**

Paper:




## Introduction
![architecture](/architecture.png)
Integrating CNNs and RNNs to capture spatiotemporal dependencies is a prevalent strategy for spatiotemporal prediction tasks. However, the property of CNNs to learn local spatial information decreases their efficiency in capturing spatiotemporal dependencies, thereby limiting their prediction accuracy. In this paper, we propose a new recurrent cell, SwinLSTM, which integrates Swin Transformer blocks and the simplified LSTM, an extension that replaces the convolutional structure in ConvLSTM with the self-attention mechanism. Furthermore, we construct a network with SwinLSTM cell as the core for spatiotemporal prediction. Without using unique tricks, SwinLSTM outperforms state-of-the-art methods on Human3.6m, TaxiBJ, KTH, and Moving MNIST datasets. In particular, it exhibits a significant improvement in prediction accuracy compared to ConvLSTM. We hope that SwinLSTM can serve as a solid baseline to promote the advancement of spatiotemporal prediction accuracy.

## Overview
- `Pretrained/` contains pretrained weights on MovingMNIST.
- `data/` contains the MNIST dataset and the compressed MovingMNIST test set.
- `SwinLSTM_B.py` contains the model with a single SwinLSTM cell.
- `SwinLSTM_D.py` contains the model with a multiple SwinLSTM cell.
- `dataset.py` contains training and testing dataloaders.
- `functions.py` contains training and testing functions.
- `main.py` is the core file for training and testing pipeline.
- `test.py` is a file for a quick test.

## Requirements
- python >=3.8
- torch
- torchvision
- numpy
- matplotlib
- skimage
- timm
- einops

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{lee2021video,
  title={Title},
  author={Author 1 and Author 2},
  booktitle={Journal},
  year={2022}
}
```

## Acknowledgment
These codes are based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We extend our sincere appreciation for their valuable contributions.

