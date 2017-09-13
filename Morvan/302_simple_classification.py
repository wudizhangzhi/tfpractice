# encoding: utf-8
from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
n_data = np.ones((100, 2))
x1 = np.random.normal(n_data * 2, 1)
y1 = np.zeros(100)

x2 = np.random.normal(n_data * -2, 1)
y2 = np.ones(100)

X = np.vstack((x1, x2))
y = np.hstack((y1, y2))

plt.scatter(X[:, 0], X[:, 1], c=y, lw=0, cmap='RdYlGn')
plt.show()


input_X = tf.placeholder(dtype=tf.float32, shape=X.shape)
input_y = tf.placeholder(dtype=tf.int32, shape=y.shape)

l1 = tf.layers.dense(input_X, 10, activation=tf.nn.relu)
output = tf.layers.dense(l1, 2)

loss = tf.losses.sparse_softmax_cross_entropy(input_y, output)
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
# accuracy = tf.reduce_mean(tf.equal(output, input_y))
accuracy = tf.metrics.accuracy(         # return (acc, update_op), and create 2 local variables
    labels=tf.squeeze(input_y), predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()
# init = tf.global_variables_initializer()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

plt.ion()

for step in xrange(100):

    _, l, acy, pred = sess.run([train_op, loss, accuracy, output], feed_dict={input_X: X, input_y: y})
    if step % 5 == 0:
        plt.cla()
        plt.scatter(X[:, 0], X[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy: %0.4f' % acy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.2)

plt.ioff()
plt.show()
