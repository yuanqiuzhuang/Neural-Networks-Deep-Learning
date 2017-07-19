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
        """初始化Network对象。列表sizes包含各层神经元数量，例如sizes = [2, 3, 1]，
        表示第一层（输入层）有2个神经元，第二层（隐藏层）有3个神经元，第三层（输出层）有
        1个神经元。网络中的偏置和权重是随机初始化的，使用np.random.randn函数生成均值
        为0，标准差为1的高斯分布。"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for (y, x) in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        """对于网络给定的输入a，返回对应的输出。"""
        for b, w in zip(self.bias, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data = None):
        """使用小批量随机梯度下降训练网络。training_data是一个(x, y)元组列表，表示
        训练输入和期望输出。epochs表示迭代次数。mini_batch_size表示小批量数据大小。
        eta是学习率learning rate。如果给出了测试数据，会在每一个迭代期后评估网络。"""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]


### 函数定义
def sigmoid(z):
    """sigmoid函数。"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prim(z):
    """sigmoid函数的导数。"""
    return sigmoid(z) * (1 - sigmoid(z))
