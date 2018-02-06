# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
#
# output = tf.multiply(input1, input2)
#
# with tf.Session() as sess:
#     result = sess.run(output, feed_dict={input1: [7], input2: [10]})
#     print(result)

def add_layer(input_x, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size], dtype=tf.float32), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('Baises'):
            baises = tf.Variable(tf.zeros(1, dtype=tf.float32), name='b')
            tf.summary.histogram(layer_name + '/baises', baises)

        with tf.name_scope('Wx_plus_b'):
            output = tf.matmul(input_x, Weights) + baises
        if activation_function is None:
            output = output
        else:
            output = activation_function(output)
        tf.summary.histogram(layer_name + '/outpts', output)
        return output

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None,1], name='y_input')

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, ys), 'float32'))

init = tf.global_variables_initializer()

sess = tf.Session()
# writer = tf.train.SummaryWriter('logs/', sess.graph)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()


for step in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys:y_data})
    if step % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, step)
        # print('accuracy: %0.2f' % sess.run(accuracy, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        y_prediction = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, y_prediction, 'r-', lw=5)
        plt.pause(0.1)
