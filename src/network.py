# -*- coding:utf-8 -*-  
__author__ = 'qzyuan'
__date__ = '2017/7/18 22:38'

import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for (y, x) in zip(sizes[1:], sizes[:-1])]

