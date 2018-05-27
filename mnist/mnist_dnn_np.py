#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/27 下午2:47
# @Author  : wudizhangzhi


import numpy as np
import tensorflow as tf


def solfmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def inference(images):
    images_reshaped = images.reshape((-1, np.prod(images.shape[1:])))

    w_1 = np.random.standard_normal((images.shape[1], 1024))
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


def train(logits, labels):
    # loss
    loss = - np.mean(np.add(np.dot(labels, np.log(logits)), np.dot(1 - labels, np.log(1 - logits))))
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
