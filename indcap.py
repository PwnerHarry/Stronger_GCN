#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:45:25 2019

@author: sitaoluan
"""
import numpy as np
from numpy.linalg import norm
from numpy import dot
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from sklearn.preprocessing import normalize

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, Dense_GCN, Simplified_GCN

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import networkx as nx
from scipy.sparse import csgraph
import math 

num_vec = 1000
num_eval = 100000
max_dim = 10
circle = 0
experiment_times = 1
activation_function = [torch.relu, F.leaky_relu, torch.selu, F.elu, torch.tanh, torch.sigmoid, F.tanhshrink, torch.hardshrink,  F.softshrink, identity]
labels = ["ReLU", "LeakyReLU", "SELU", "ELU", "TanH", "Sigmoid", "Tanhshrink", "HardShrink",  "Softshrink", "Identity"]



angle_result = np.matrix(np.zeros([len(activation_function), experiment_times]), dtype = np.float64)
angle_all = np.matrix(np.zeros([len(activation_function), max_dim - 1]), dtype = np.float64)
std_all = np.matrix(np.zeros([len(activation_function), max_dim - 1]), dtype = np.float64)


for ndim in range(2, max_dim+1):

    for activation, label, j in zip(activation_function, labels, range(len(activation_function))):
        
        angle_result = np.matrix(np.zeros([len(activation_function), experiment_times]))
        
        for i in range(experiment_times):
            
            # Generate points
            if circle == 1:
                vec = np.random.randn(ndim, num_vec)
                vec_norm = np.linalg.norm(vec, axis=0) 
                vec_norm[vec_norm == 0] = 1e-8
                vec /= vec_norm
                vec_length = np.random.uniform(0,10, (1,num_vec))
                vec *= vec_length
            else:
                vec = np.random.uniform(0,1, (ndim, num_vec))
                
            
            vec = activation(torch.tensor(vec))
            vec = vec.numpy()
            vec_norm = np.linalg.norm(vec, axis=0) 
            vec_norm[vec_norm == 0] = 1e-8
            vec /= vec_norm
            # Do evaluation for each point
            #for j in range(num_vec):
            if circle == 1:
                vec_eval = np.random.randn(ndim, num_eval)
                vec_eval_norm = np.linalg.norm(vec_eval, axis=0) 
                vec_eval_norm[vec_eval_norm == 0] = 1e-8
                vec_eval /= vec_eval_norm
                vec_eval_length = np.random.uniform(0,10, (1,num_eval))
                vec_eval *= vec_eval_length
            else: 
                vec_eval = np.random.uniform(0,1, (ndim, num_eval))
            
            vec_eval = activation(torch.tensor(vec_eval))
            vec_eval = vec_eval.numpy()
            vec_eval_norm = np.linalg.norm(vec_eval, axis=0) 
            vec_eval_norm[vec_eval_norm == 0] = 1e-8
            vec_eval /= vec_eval_norm
            
            vec_angles = np.transpose(vec) @ vec_eval
            angle_result[j,i] = np.sum(vec_angles>(1-1e-8)) #/(num_vec*num_eval)
            print(label, np.sum(vec_angles>(1-1e-8))/(num_vec*num_eval), angle_result[j,i])
           
            
        angle_all[:, ndim-2] = np.mean(angle_result, 1)
        std_all[:, ndim-2] = np.std(angle_result, 1)













def identity(x):
    return x