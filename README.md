# Snowball and Truncated Krylov Graph Convolutional Networks

PyTorch and TensorFlow2 implementation of Snowball and Truncated Krylov Graph Convolutional Network (GCN) architectures for semi-supervised classification [1].

This repository contains the Cora, CiteSeer and PubMed dataset.

## Performance Ranking

Results are collected through the PyTorch implementation. These results WILL BE UPDATED since we have greatly optimized the implementations.

There are slight differences between the 2 implementations, so you may have to redo the hyperparameter search for the TensorFlow2 implementation.

Please feel free to leave comments if you have trouble reproducing the results!

### Cora
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-cora-05)](https://paperswithcode.com/sota/node-classification-on-cora-05?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-cora-1)](https://paperswithcode.com/sota/node-classification-on-cora-1?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-cora-3)](https://paperswithcode.com/sota/node-classification-on-cora-3?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-cora-with-public-split)](https://paperswithcode.com/sota/node-classification-on-cora-with-public-split?p=break-the-ceiling-stronger-multi-scale-deep)

### CiteSeer
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-citeseer-05)](https://paperswithcode.com/sota/node-classification-on-citeseer-05?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-citeseer-1)](https://paperswithcode.com/sota/node-classification-on-citeseer-1?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-citeseer-with-public)](https://paperswithcode.com/sota/node-classification-on-citeseer-with-public?p=break-the-ceiling-stronger-multi-scale-deep)

### PubMed
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-pubmed-003)](https://paperswithcode.com/sota/node-classification-on-pubmed-003?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-pubmed-005)](https://paperswithcode.com/sota/node-classification-on-pubmed-005?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-pubmed-01)](https://paperswithcode.com/sota/node-classification-on-pubmed-01?p=break-the-ceiling-stronger-multi-scale-deep)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/break-the-ceiling-stronger-multi-scale-deep/node-classification-on-pubmed-with-public)](https://paperswithcode.com/sota/node-classification-on-pubmed-with-public?p=break-the-ceiling-stronger-multi-scale-deep)

## Requirements

  * PyTorch 1.3.x or TensorFlow 2.x.x
  * Python 3.6+
  * Best with NVIDIA apex (we have used the NGC container with singularity)

## Initialization

```python initialize_dataset.py```

## Usage

```python train.py```

## References

[1] [Luan, et al., Break the Ceiling: Stronger Multi-scale Deep Graph Convolutional Networks, 2019](https://arxiv.org/abs/1906.02174)

## Cite

Please kindly cite our work if necessary:

```
@incollection{luan2019break,
title = {Break the Ceiling: Stronger Multi-scale Deep Graph Convolutional Networks},
author = {Luan, Sitao and Zhao, Mingde and Chang, Xiao-Wen and Precup, Doina},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {10943-10953},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {https://arxiv.org/abs/1906.02174}
}
```