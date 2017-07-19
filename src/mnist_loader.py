# -*- coding:utf-8 -*-  
__author__ = 'qzyuan'
__date__ = '2017/7/19 12:01'

import cPickle
import gzip

import numpy as np

def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    X_train = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    Y_train = [vectorized_results(y) for y in tr_d[1]]
    training_data = zip(X_train, Y_train)
    X_val = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(X_val, va_d[1])
    X_test = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(X_test, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_results(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
