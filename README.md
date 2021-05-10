# TrSeg: Transformer for Semantic Segmentation

### Introduction

This repository is a PyTorch implementation of [TrSeg](https://hszhao.github.io/projects/psanet). This work is based on [semseg](https://github.com/hszhao/semseg/blob/1.0.0/README.md).

<img src="./figure/TrSeg.png" width="900"/>

The code is easy to use for training and testing on various datasets. The codebase mainly uses ResNet50/101/152 as backbone and can be easily adapted to other basic classification structures. Implemented networks including [PSPNet](https://hszhao.github.io/projects/pspnet) and [PSANet](https://hszhao.github.io/projects/psanet), which ranked 1st places in [ImageNet Scene Parsing Challenge 2016 @ECCV16](http://image-net.org/challenges/LSVRC/2016/results), [LSUN Semantic Segmentation Challenge 2017 @CVPR17](https://blog.mapillary.com/product/2017/06/13/lsun-challenge.html) and [WAD Drivable Area Segmentation Challenge 2018 @CVPR18](https://bdd-data.berkeley.edu/wad-2018.html). Sample experimented datasets are [ADE20K](http://sceneparsing.csail.mit.edu), [PASCAL VOC 2012](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6) and [Cityscapes](https://www.cityscapes-dataset.com).
