# encoding: utf-8

from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.data import Dataset

# fake data
npx = np.random.uniform(-1, 1, (1000, 1))
npy = np.power(npx, 2) + np.random.normal(0, 0.1, size=npx.shape)

npx_train, npx_test = np.split(npx, [800])
npy_train, npy_test = np.split(npy, [800])

tfx = tf.placeholder(tf.float32, npx_train.shape)
tfy = tf.placeholder(tf.float32, npy_train.shape)

# create dataloader
dataset = Dataset.from_tensor_slices((tfx, tfy))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.repeat(3)
iterator = dataset.make_initializable_iterator()

# network
bx, by = iterator.get_next()
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, npy.shape[1])
loss = tf.losses.mean_squared_error(by, out)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# need to initialize the iterator in this case
sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tfx: npx_train, tfy: npy_train})

for step in xrange(201):
    try:
        _, trainl = sess.run([train, loss])
        if step % 10 == 0:
            testl = sess.run(loss, {bx: npx_test, by: npy_test})
            print('step : %i/200' % step, '|trainl: %0.4f' % trainl, '|testl: %0.4f' % testl)

    except tf.errors.OutOfRangeError:
        print('Finish last epoch')
        break
