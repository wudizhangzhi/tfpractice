# -*- coding:utf8 -*-

import tensorflow as tf
from  sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

# load data
digits = load_digits()
X = digits.data
y = digits.target

y = LabelBinarizer().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

def add_layer(input_x, in_size, out_size, layer_n, activation_function=None):
    layer_name ='layer_%s' % layer_n
    with tf.name_scope(layer_name):
        with tf.name_scope('W'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size] ,dtype=tf.float32), name='Weights')

        with tf.name_scope('Baises'):
            baises = tf.Variable(tf.zeros(1, dtype=tf.float32), name='baises')

        Wx_plus_baises = tf.matmul(input_x, Weights) + baises
        # drop out
        Wx_plus_baises = tf.nn.dropout(Wx_plus_baises, keep_prob)

        if activation_function is not None:
            output = activation_function(Wx_plus_baises)
        else:
            output = Wx_plus_baises
        # TODO
        return output


# define placeholder
keep_prob = tf.placeholder(dtype=tf.float32)
xs = tf.placeholder(dtype=tf.float32, shape=[None, 8*8])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 10])


# add layer
l1 = add_layer(xs, 8*8, 100, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 100, 10, 'output', activation_function=tf.nn.softmax)

# loss
loss = tf.reduce_mean(tf.reduce_sum(-ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', loss)

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# init
sess = tf.Session()

merged = tf.summary.merge_all()
writer_train = tf.summary.FileWriter('logs/train', graph=sess.graph)
writer_test = tf.summary.FileWriter('logs/test', graph=sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for step in range(500):
    sess.run(train, feed_dict={xs: X_train, ys: y_train, keep_prob: 1.0})
    if step % 50 == 0:
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1.0})
        # record loss
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0})
        writer_train.add_summary(train_result, step)
        writer_test.add_summary(test_result, step)
