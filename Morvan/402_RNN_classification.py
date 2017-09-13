# encoding: utf-8


from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 35
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01

# data
mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot some data

# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title(mnist.train.labels[0])
# plt.show()


tf_x = tf.placeholder(dtype=tf.float32, shape=[None, TIME_STEP * INPUT_SIZE])  # [batch_size, step * input_size]
images = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])  # [batch_size, step, input_size]
tf_y = tf.placeholder(dtype=tf.int32, shape=[None, 10])  # [bathch, label]

# RNN cell
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
    num_units=64,
    forget_bias=1.0,
    state_is_tuple=True,
    activation=None,
    reuse=None
)

outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,
    images,
    initial_state=None,
    dtype=tf.float32,
    time_major=False,  # False [batch, time_step, input]; True [time_step, bathc, input]
)

output = tf.layers.dense(outputs[:, -1, :], 10)  # output base on the last output step

losses = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)

train_op = tf.train.AdamOptimizer(LR).minimize(losses)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())  # the local variable is accuracy op
sess.run(init_op)

for step in xrange(1000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, _l = sess.run([train_op, losses], feed_dict={tf_x: b_x, tf_y: b_y})

    if step % 50 == 0:
        _ac = sess.run(accuracy, feed_dict={tf_x: test_x, tf_y: test_y})
        print('loss: %0.4f,  accuracy: %0.4f' % (_l, _ac))


# print 10 predict data
p_x = test_x[:10]
p_y = sess.run(output, feed_dict={tf_x: p_x})

plt.figure()
for i in xrange(p_x.shape[0]):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(p_x[i, :], [28, 28]), cmap='gray')
    plt.title(np.argmax(p_y[i, :]))
plt.show()
