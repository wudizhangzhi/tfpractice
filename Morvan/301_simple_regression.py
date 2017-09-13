# encoding: utf-8
from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(1)
np.random.seed(1)

# fake data
X = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=X.shape)

y = np.power(X, 2) + noise

# show data
# plt.scatter(X, y)
# plt.show()


tf_x = tf.placeholder(dtype=tf.float32, shape=X.shape)
tf_y = tf.placeholder(dtype=tf.float32, shape=y.shape)


# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 1)

loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting
for step in xrange(100):
    _, l, pred = sess.run([optimizer, loss, output], feed_dict={tf_x: X, tf_y: y})
    if step % 5 == 0:
        plt.cla()
        plt.scatter(X, y)
        plt.plot(X, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'loss:%0.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.2)

plt.ioff()
plt.show()




