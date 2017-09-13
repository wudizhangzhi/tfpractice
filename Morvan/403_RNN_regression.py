# encoding: utf-8

from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

CELL_SIZE = 32
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.01

# steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, x_np, 'r-', label='input sin')
# plt.plot(steps, y_np, 'b--', label='target cos')
# plt.show()

# tensorflow placeholder
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])  # shape [batch, step, inputsize]
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
    num_units=CELL_SIZE,
    forget_bias=1.0,
    state_is_tuple=True,
    activation=None,
    reuse=None
)
# rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)

init_s = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,
    inputs=tf_x,
    initial_state=init_s,
    time_major=False,
)

outs2D = tf.reshape(outputs, [-1, CELL_SIZE])
net_outs2D = tf.layers.dense(inputs=outs2D, units=INPUT_SIZE)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])  # reshape back ot 3D

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.figure(1, figsize=(12, 5))
plt.ion()
for step in xrange(60):
    start, end = step * np.pi, (step + 1) * np.pi  # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP)
    x = np.sin(steps)[np.newaxis, :, np.newaxis]  # shape to [batch, step, input_size]
    y = np.cos(steps)[np.newaxis, :, np.newaxis]

    if 'final_s_' in globals():
        feed_dict = {tf_x: x, tf_y: y, init_s: final_s_}
    else:
        feed_dict = {tf_x: x, tf_y: y}

    _, pred_, final_s_, loss_ = sess.run([train_op, outs, final_s, loss], feed_dict=feed_dict)

    print(loss_)

    plt.plot(steps, y.flatten(), 'r-')
    plt.plot(steps, pred_.flatten(), 'b-')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.3)

plt.ioff()
plt.show()
