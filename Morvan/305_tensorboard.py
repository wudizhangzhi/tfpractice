# encoding: utf-8

from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
X = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=X.shape)
y = np.power(X, 2) + noise

with tf.variable_scope('Input'):
    input_X = tf.placeholder(dtype=tf.float32, shape=X.shape, name='x')
    input_y = tf.placeholder(dtype=tf.float32, shape=y.shape, name='y')

with tf.variable_scope('Net'):
    l1 = tf.layers.dense(input_X, 10, activation=tf.nn.relu)
    output = tf.layers.dense(l1, 1)

    tf.summary.histogram('h_out', l1)
    tf.summary.histogram('pred', output)

loss = tf.losses.mean_squared_error(input_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)  # add loss to scalar summary

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./log', sess.graph)  # write to file
merged = tf.summary.merge_all()  # operation to merge all summary


for step in xrange(100):
    _, result = sess.run([train_op, merged], feed_dict={input_X: X, input_y: y})
    writer.add_summary(result, step)
