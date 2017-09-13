# encoding: utf-8

from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data

LR = 0.02
BATCH_SIZE = 64
N_TEST_IMG = 5

# data
mnist = input_data.read_data_sets('./mnist', one_hot=False)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])

# encoder
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
en3 = tf.layers.dense(en2, 3)

# decoder
de0 = tf.layers.dense(en3, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(en1, 128, tf.nn.tanh)
de3 = tf.layers.dense(en2, 28 * 28)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=de3)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

f, a = plt.subplots(2, N_TEST_IMG, figsize=(2, 5))
plt.ion()

view_data = test_x[:N_TEST_IMG, :]
for i in xrange(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for step in xrange(8000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, _encoder, _decoder, _loss = sess.run([train_op, en3, de3, loss], feed_dict={tf_x: b_x})

    if step % 20 == 0:
        print('loss: %0.4f' % _loss)
        tmp_decoder = sess.run(de3, feed_dict={tf_x: view_data})
        for i in xrange(len(tmp_decoder)):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(tmp_decoder[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.01)

plt.ioff()

# visualize in 3D
view_data = test_x[:200]
fig = plt.figure(2)
ax = Axes3D(fig)

_encoder = sess.run(en3, feed_dict={tf_x: view_data})
X, Y, Z = _encoder[:, 0], _encoder[:, 1], _encoder[:, 2]
for x, y, z, s in zip(X, Y, Z, test_y):
    c = cm.rainbow(int(255 * s / 9))
    ax.text(x, y, z, s, backgroundcolor=c)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
