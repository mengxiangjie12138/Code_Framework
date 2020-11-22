import torch
import numpy as np
import os
import sklearn.cluster
from networks_and_related_funs.ClassicNetwork.ResNet import ResNet50
import matplotlib.pyplot as plt

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


if __name__ == '__main__':
    a = np.load(r'D:\My_Code\mengxiangjie\DTGCN\evaluation2\resnet18_binary_features\label_true.npy')
    print(len(a))



