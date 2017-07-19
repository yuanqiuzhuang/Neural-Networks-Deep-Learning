# -*- coding:utf-8 -*-  
__author__ = 'qzyuan'
__date__ = '2017/7/19 14:30'

import mnist_loader
import network

train_data, val_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 50, 10])
net.SGD(train_data, 30, 20, 3.0, test_data=test_data)

"""
测试结果：
Epoch 0: 8901 / 10000
Epoch 1: 9211 / 10000
Epoch 2: 9294 / 10000
Epoch 3: 9331 / 10000
Epoch 4: 9405 / 10000
Epoch 5: 9386 / 10000
Epoch 6: 9396 / 10000
Epoch 7: 9404 / 10000
Epoch 8: 9452 / 10000
Epoch 9: 9427 / 10000
Epoch 10: 9451 / 10000
Epoch 11: 9474 / 10000
Epoch 12: 9485 / 10000
Epoch 13: 9483 / 10000
Epoch 14: 9494 / 10000
Epoch 15: 9493 / 10000
Epoch 16: 9471 / 10000
Epoch 17: 9493 / 10000
Epoch 18: 9490 / 10000
Epoch 19: 9513 / 10000
Epoch 20: 9506 / 10000
Epoch 21: 9513 / 10000
Epoch 22: 9506 / 10000
Epoch 23: 9519 / 10000
Epoch 24: 9512 / 10000
Epoch 25: 9514 / 10000
Epoch 26: 9530 / 10000
Epoch 27: 9517 / 10000
Epoch 28: 9509 / 10000
Epoch 29: 9523 / 10000
"""