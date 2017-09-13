# encoding: utf-8

from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
X = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, (100, 1))
y = np.power(X, 2) + noise


def save():
    input_X = tf.placeholder(dtype=tf.float32, shape=X.shape)
    input_y = tf.placeholder(dtype=tf.float32, shape=y.shape)

    l1 = tf.layers.dense(input_X, 10, activation=tf.nn.relu)  # hidden
    output = tf.layers.dense(l1, 1)

    loss = tf.losses.mean_squared_error(labels=input_y, predictions=output)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()  # define a saver

    for step in xrange(100):
        _, l = sess.run([train_op, loss], feed_dict={input_X: X, input_y: y})

    saver.save(sess, 'params', write_meta_graph=False)  # meta_graph not recommended

    # ploting
    pred, l = sess.run([output, loss], feed_dict={input_X: X, input_y: y})
    plt.figure(1, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X, y)
    plt.plot(X, pred, 'r-', lw=2)
    plt.text(-1, 1.2, 'Save Loss: %0.4f' % l, fontdict={'size': 15, 'color': 'red'})

def reload():
    input_X = tf.placeholder(dtype=tf.float32, shape=X.shape)
    input_y = tf.placeholder(dtype=tf.float32, shape=y.shape)

    l1 = tf.layers.dense(input_X, 10, activation=tf.nn.relu)
    output = tf.layers.dense(l1, 1)

    loss = tf.losses.mean_squared_error(labels=input_y, predictions=output)

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, 'params')

    pred, l = sess.run([output, loss], feed_dict={input_X: X, input_y: y})

    plt.subplot(1, 2, 2)
    plt.scatter(X, y)
    plt.plot(X, pred, 'r-', lw=2)
    plt.text(-1, 1.2, 'Loss: %0.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()


save()

# destory previous graph
tf.reset_default_graph()

reload()
