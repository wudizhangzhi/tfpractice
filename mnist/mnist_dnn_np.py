#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/27 下午2:47
# @Author  : wudizhangzhi


import numpy as np
import tensorflow as tf


class FullConnectLayer(object):
    def __init__(self, inputs, units):
        self._inputs = inputs
        self._units = units


def solfmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def sigmod(x):
    return (1 + np.exp(-x)) ** -1


def inference(images):
    images_reshaped = images.reshape((-1, np.prod(images.shape[1:])))

    w_1 = np.random.standard_normal((images_reshaped.shape[1], 1024))
    b_1 = np.zeros_like((images.shape[0],))
    dense_1 = np.add(np.matmul(images, w_1), b_1)
    # relu
    dense_1 = np.maximum(dense_1, 0)

    w_2 = np.random.standard_normal((dense_1.shape[1], 256))
    b_2 = np.zeros_like((dense_1.shape[0],))
    dense_2 = np.add(np.matmul(dense_1, w_2), b_2)
    # relu
    dense_2 = np.maximum(dense_2, 0)

    w_3 = np.random.standard_normal((dense_2.shape[1], 10))
    b_3 = np.zeros_like((dense_2.shape[0],))
    dense_3 = np.add(np.matmul(dense_2, w_3), b_3)
    # solfmax
    logits = solfmax(dense_3)
    return logits


def predict(features, weights):
    return


def update_weight(features, labels, weights, lr):
    """
    :param features: [batch_size, units]
    :param labels: [batch_size, classes]
    :param weights: [units, classes]
    :param lr: learning rate
    :return:
    """
    # TODO 最后一层
    predicts = predict(features, weights)  # [batch_size, classes]
    gradient = np.dot(features.T, predicts - labels)  # [units, batch_size] * [batch_size, classes]

    gradient /= len(features)

    gradient *= lr

    weights -= gradient
    return weights


def train(features, labels, iters):
    logistic = inference(features)
    # loss
    loss = - np.mean(np.add(np.dot(labels, np.log(logistic)), np.dot(1 - labels, np.log(1 - logistic))))
    # gradient

    # train_op

    pass


def main():
    # load data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


if __name__ == '__main__':
    main()
