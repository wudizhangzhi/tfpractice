# encoding: utf-8

from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32


# fake data
X = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=X.shape)
y = np.power(X, 2) + noise



class Net:
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(dtype=tf.float32, shape=X.shape)
        self.y = tf.placeholder(dtype=tf.float32, shape=y.shape)

        self.l1 = tf.layers.dense(self.x, 10, activation=tf.nn.relu)
        self.output = tf.layers.dense(self.l1, 1)

        self.loss = tf.losses.mean_squared_error(self.y, self.output)
        self.train = opt(LR, **kwargs).minimize(self.loss)

# different nets
net_SGD         = Net(tf.train.GradientDescentOptimizer)
net_Momentum    = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop     = Net(tf.train.RMSPropOptimizer)
net_Adam        = Net(tf.train.AdamOptimizer)

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

losses_his = [[], [], [], []]   # record loss

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in xrange(300):
    indexes = np.random.randint(0, X.shape[0], BATCH_SIZE)
    b_x = X[indexes]
    b_y = y[indexes]

    for net, l_his in zip(nets, losses_his):
        _, l = sess.run([net.train, net.loss], feed_dict={net.x: X, net.y: y})
        l_his.append(l)


# plot loss history
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])

plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
