# -*- coding:utf-8 -*-  
__author__ = 'qzyuan'
__date__ = '2017/7/18 22:38'

"""
一个实现前馈神经网络的sgd学习算法模块。使用反向传播计算梯度。
"""

import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for (y, x) in zip(sizes[1:], sizes[:-1])]
    

### 函数定义
def sigmoid(z):
    """sigmoid函数。"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prim(z):
    """sigmoid函数的导数。"""
    return sigmoid(z) * (1 - sigmoid(z))
