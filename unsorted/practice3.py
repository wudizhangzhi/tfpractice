# -*- coding:utf8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(input_x, in_size, out_size, layer_n, activation_function=None):
    layer_name ='layer_%s' % layer_n
    with tf.name_scope(layer_name):
        with tf.name_scope('W'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size] ,dtype=tf.float32), name='Weights')

        with tf.name_scope('Baises'):
            baises = tf.Variable(tf.zeros(1, dtype=tf.float32), name='baises')

        Wx_plus_baises = tf.matmul(input_x, Weights) + baises

        if activation_function is not None:
            output = activation_function(Wx_plus_baises)
        else:
            output = Wx_plus_baises
        return output

with tf.name_scope('Input'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name='X')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

# input layer
# with tf.name_scope('Prediction'):
prediction = add_layer(xs, 28*28, 10, 1, activation_function=tf.nn.softmax)


# loss
with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('cross_entopy', cross_entropy)
# train
train = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

# accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    with tf.name_scope('Accuracy'):
        y_pre = sess.run(prediction, feed_dict={xs: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        tf.summary.scalar('accuracy', accuracy)
    return result


# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', graph=sess.graph)



# start
for step in range(500):
    with tf.name_scope('Train'):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
    if step % 50 == 0:
        accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
        result = sess.run(merged, feed_dict={xs: mnist.test.images, ys: mnist.test.labels})
        writer.add_summary(result, step)
        print('%0.2f' % accuracy)
