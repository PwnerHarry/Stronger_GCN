# Snowball and Truncated Krylov Graph Convolutional Networks in PyTorch

PyTorch implementation of Snowball and Truncated Krylov Graph Convolutional Network (GCN) architectures for semi-supervised classification [1].

This repository contains the Cora, CiteSeer and PubMed dataset.

## Performance Ranking

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

  * PyTorch 1.0.0+
  * Python 3.6+
  * Best with CUDA/10.0 and NVIDIA apex (we have used the container docker://nvcr.io/nvidia/pytorch:19.05-py3 with singularity)

## Initialization

```python initialize_dataset.py```

## Usage

```python train.py```

## References

[1] [Luan, et al., Break the Ceiling: Stronger Multi-scale Deep Graph Convolutional Networks, 2019](https://arxiv.org/abs/1906.02174)

## Cite

Please kindly cite our work if necessary:

```
@article{luan2019break,
title={Break the Ceiling: Stronger Multi-scale Deep Graph Convolutional Networks},
author={Luan, Sitao and Zhao, Mingde and Chang, Xiao-Wen and Precup, Doina},
journal={arXiv},
volume={1906.02174},
year={2019},
url={https://arxiv.org/abs/1906.02174},
}
```
