# -*- coding:utf8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#######################################
### CNN Convolutional Nerul Network ###
#######################################

# load data
mnist = input_data.read_data_sets('MINIST_data', one_hot=True)


## functions ##
def compute_accuracy(X, y):
    global prediction
    y_pred = sess.run(prediction, feed_dict={xs: X, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.arg_max(ys, 1), tf.arg_max(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: X, ys: y, keep_prob:1.0})
    tf.summary.scalar('accuracy', result)
    return result

def weight_variable(shape):
    '''
    shape: [filter_height, filter_width, in_channel, out_channel]
    '''
    init = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init)

def baises_variable(shape):
    init = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init)

def conv2d(x, W):
    '''
    strides: A list of `ints`.
        1-D tensor of length 4.  The stride of the sliding window for each
        dimension of `input`.
        strides = [1, x_movement , y_movement, 1]
        must strides[0] = strides[3] = 1
    padding: 'SAME' or 'VALID'
    date_format: 'NHWC': [batch, height, width, channel]
                 'NCHW': [batch, channel, height, width]
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    ksize: A list of ints that has length >= 4.  The size of the window for
        each dimension of the input tensor.
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define placeholder ##
xs = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])/255.
ys = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32]) # filter size: 5*5, in_channel=1因为是黑白的, out_chennel设置为32
b_conv1 = baises_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size: 28*28*32
h_pool1 = max_pool_2x2(h_conv1) #  以为横纵步长为2, output size: 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = baises_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size: 14*14*64
h_pool2 = max_pool_2x2(h_conv2) # output size: 7*7*64

## func1 layer ##
# [n_sample, h, w, channel] ==> [n_sample, h*w*channel]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = baises_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer(output layer) ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = baises_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

## loss ##
cross_entropy = tf.reduce_mean(tf.reduce_sum(-ys*tf.log(prediction), axis=1))
tf.summary.scalar('cross_entropy', cross_entropy)
# train = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

## init ##
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## summary ##
merged = tf.summary.merge_all()
writer_train = tf.summary.FileWriter('logs/tarin', sess.graph)
writer_test = tf.summary.FileWriter('logs/test', sess.graph)

for step in range(500):
    batch_X, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={xs: batch_X, ys:batch_y, keep_prob: 0.5})
    if step % 50 == 0:
        # for print
        # loss = sess.run(cross_entropy, feed_dict={xs: batch_X, ys:batch_y, keep_prob: 0.5})
        # print('loss: %0.2f' % loss)
        test_X, test_y = mnist.test.images, mnist.test.labels
        test_accuracy = compute_accuracy(test_X, test_y)
        print('accuracy: %0.4f' % test_accuracy)
        # for summary
        summary_part = sess.run(merged, feed_dict={xs: test_X, ys:test_y, keep_prob: 1.0})
        writer_test.add_summary(summary_part, step)
