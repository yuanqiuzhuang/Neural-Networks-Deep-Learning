# -*- coding:utf-8 -*-  
__author__ = 'qzyuan'
__date__ = '2017/7/19 14:30'

import mnist_loader
import network

train_data, val_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(train_data, 30, 10, 3.0, test_data=test_data)
