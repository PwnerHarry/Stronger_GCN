#!/bin/bash

module load singularity
singularity pull docker://nvcr.io/nvidia/pytorch:19.05-py3
singularity exec --nv ~/pytorch-19.05-py3.simg python initialize_dataset.py