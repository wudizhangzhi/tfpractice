# encoding: utf-8

"""Builds the MNIST network.
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""

import tensorflow as tf
import numpy as np

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIEXLS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_unit, hidden2_unit):
    # hidden1
    with tf.name_scope('hidden1'):
        Weights = tf.Variable(tf.truncated_normal(shape=[IMAGE_PIEXLS, hidden1_unit],
                                                  stddev=1.0 / np.sqrt(float(IMAGE_PIEXLS))), name='wieghts')
        biases = tf.Variable(tf.zeros(shape=[hidden1_unit]), name='biases')

        hidden1 = tf.nn.bias_add(tf.matmul(images, Weights), biases)

    # hidden2
    with tf.name_scope('hidden2'):
        Weights = tf.Variable(tf.truncated_normal(shape=[hidden1_unit, hidden2_unit],
                                                  stddev=1.0 / np.sqrt(float(hidden1_unit))), name='wieghts')
        biases = tf.Variable(tf.zeros(shape=[hidden2_unit]), name='biases')

        hidden2 = tf.nn.bias_add(tf.matmul(hidden1, Weights), biases)

    # linear
    with tf.name_scope('softmax_linear'):
        Weights = tf.Variable(tf.truncated_normal(shape=[hidden2_unit, NUM_CLASSES],
                                                  stddev=1.0 / np.sqrt(float(hidden2_unit))), name='wieghts')
        biases = tf.Variable(tf.zeros(shape=[NUM_CLASSES]), name='biases')

        logits = tf.nn.bias_add(tf.matmul(hidden2, Weights), biases)

    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evalution(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
