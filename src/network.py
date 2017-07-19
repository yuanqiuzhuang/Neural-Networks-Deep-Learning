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
                training_data[k : k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
                )
            else:
                print("Epoch {0} complete.".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """使用mini_batch中的训练数据，根据单次梯度下降的迭代更新网络的偏置和权重。"""
        # 初始化每一层的偏置和权重梯度。
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # 调用反向传播算法，快速计算损失函数的梯度。
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - eta / len(mini_batch) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b - eta / len(mini_batch) * nb
                     for b, nb in zip(self.bias, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """网络评估函数。返回神经网络正确预测的测试数据的数目。"""
        # 前馈网络计算得到的10维向量值最大的下标即预测结果。
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)



### 函数定义
def sigmoid(z):
    """sigmoid函数。"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prim(z):
    """sigmoid函数的导数。"""
    return sigmoid(z) * (1 - sigmoid(z))

"""
sizes = [2, 3, 1]
net = Network(sizes)
print net.weights
print net.bias
"""
