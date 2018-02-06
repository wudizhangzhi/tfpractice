# -*- coding:utf8 -*-

from __future__ import print_function
from __future__ import absolute_import

from six.moves import xrange
from six.moves import urllib
import sys
import os
import gzip
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

# parameters
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
# IMAGE_SIZE = 28
IMAGE_SIZE = 40
# NUM_CHANNELS = 1
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
LABEL = 10
KEEP_PROB = 0.5
FLAGS = None
STDDEV = 0.1
SEED = 1000
NUM_EPOCHS = 10
BATCH_SIZE = 100
EVAL_FREQUENCY = 100
EVAL_BATCH_SIZE = 64
# VALIDATION_SIZE = 5000
VALIDATION_SIZE = 2000


def get_type():
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE *
                              num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def conv2d(x, W):
    '''
    x: [batch, in_height, in_width, in_channel]
    W: [filter_height, filter_width, out_channel, in_channel]
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(value):
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=STDDEV, seed=SEED, dtype=get_type())
    return tf.Variable(init)


def biases_variable(shape):
    init = tf.constant(0.1, shape=shape, dtype=get_type())
    return tf.Variable(init)


def accuracy_rate(predictions, labels):
    # return 100.0 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, axis=1), labels), tf.float32))
    return 100.0 * np.sum(np.argmax(predictions, axis=1) == labels) / predictions.shape[0]


def plotrandom(images, labels, num=8, image_size=28, channel=1):
    for i in range(num):
        index = np.random.randint(len(images))
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        if channel == 1:
            plt.imshow(images[index].reshape(image_size, image_size), cmap=plt.cm.gray_r, interpolation='nearest')
        else:
            plt.imshow(images[index].reshape(image_size, image_size, channel), cmap=plt.cm.gray_r,
                       interpolation='nearest')
        # plt.title('Training: %s' % labels[index].tolist().index(1))
        plt.title('Training: %s' % labels[index])
    plt.show()


def main(_):
    #### load data ####
    # train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    # train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    # test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    # test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    # # Extract it into np arrays.
    # train_data = extract_data(train_data_filename, 60000)
    # train_labels = extract_labels(train_labels_filename, 60000)
    # test_data = extract_data(test_data_filename, 10000)
    # test_labels = extract_labels(test_labels_filename, 10000)
    X = np.load('../X_train.npy')
    y = np.load('../y_train.npy')
    train_data = X[::2, :]
    train_labels = y[::2, :]
    train_data = train_data.reshape([-1, 40, 40, 3])

    validation_data = X[1::2, :]
    validation_data = validation_data.reshape([-1, 40, 40, 3])
    validation_labels = y[1::2, :]
    # validation_data = train_data[:VALIDATION_SIZE, ...]
    # validation_labels = train_labels[:VALIDATION_SIZE]
    # train_data = train_data[VALIDATION_SIZE:, ...]
    # train_labels = train_labels[VALIDATION_SIZE:]

    # test_index = np.random.randint(0, 10000, size=EVAL_BATCH_SIZE)
    # test_data = test_data[test_index,:]
    # test_labels = test_labels[test_index]

    #### placeholder ####
    train_data_node = tf.placeholder(
        shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], dtype=get_type())
    # TODO
    train_label_node = tf.placeholder(shape=[BATCH_SIZE, LABEL], dtype=tf.int64)
    keep_prob = tf.placeholder(dtype=get_type())
    eval_data = tf.placeholder(
        get_type(),
        shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_size = train_labels.shape[0]  # 训练样本总数

    # variables
    w_conv1 = weight_variable([5, 5, NUM_CHANNELS, 32])
    # bias_conv1=biases_variable([32])
    bias_conv1 = tf.Variable(tf.zeros([32], dtype=get_type()))

    w_conv2 = weight_variable([5, 5, 32, 64])
    bias_conv2 = biases_variable([64])

    w_fc1 = weight_variable([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512])
    bias_fc1 = biases_variable([512])

    w_fc2 = weight_variable([512, LABEL])
    bias_fc2 = biases_variable([10])

    def model(data, is_train=False):  # create layer
        ######## layer 1 ########
        # conv1
        data_conv1 = conv2d(data, w_conv1)
        relu_conv1 = tf.nn.relu(tf.nn.bias_add(data_conv1, bias_conv1))
        # pool1
        pool_conv1 = max_pool_2x2(relu_conv1)
        ######## layer 1 ########
        # conv2
        data_conv2 = conv2d(pool_conv1, w_conv2)
        relu_conv2 = tf.nn.relu(tf.nn.bias_add(data_conv2, bias_conv2))
        relu_conv2 = tf.clip_by_value(relu_conv2, 1e-10, 1.)
        # pool2
        pool_conv2 = max_pool_2x2(relu_conv2)
        ######## hidden layer 1 ########
        # full connect hidden layer
        pool_shape = pool_conv2.get_shape().as_list()
        reshape = tf.reshape(pool_conv2,
                             [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        data_hidden = tf.nn.relu(tf.matmul(reshape, w_fc1) + bias_fc1)
        data_hidden = tf.clip_by_value(data_hidden, 1e-10, 1.)
        if is_train:
            data_hidden = tf.nn.dropout(data_hidden, KEEP_PROB, seed=SEED)
        ######## output layer ########
        # return tf.matmul(data_hidden, w_fc2) + bias_fc2
        return tf.clip_by_value(tf.matmul(data_hidden, w_fc2) + bias_fc2, 1e-10, 1.)

    #### create layer ####
    logits = model(train_data_node, is_train=True)
    #### loss ####
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=train_label_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(bias_fc1) +
                    tf.nn.l2_loss(w_fc2) + tf.nn.l2_loss(bias_fc2))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    tf.summary.scalar('loss', loss)
    #### train ####
    # TODO
    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=get_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    # TODO
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    train_prediction = tf.nn.softmax(logits)

    train_eval = tf.nn.softmax(model(eval_data))

    def eval_in_batches(data, sess):
        '''get all prediction by run a small batch data'''
        eval_size = data.shape[0]
        if eval_size < EVAL_BATCH_SIZE:
            raise ValueError('eval_size must larger than EVAL_BATCH_SIZE')
        predictions = np.ndarray(shape=(eval_size, LABEL), dtype='float32')
        for begin in xrange(0, eval_size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= eval_size:
                predictions[begin:end, :] = sess.run(train_eval, feed_dict={
                    eval_data: data[begin:end, ...]
                })
            else:
                batch_predictions = sess.run(train_eval, feed_dict={
                    eval_data: data[-EVAL_BATCH_SIZE:, ...]
                })
                predictions[begin:, :] = batch_predictions[begin - eval_size, :]
        return predictions

    #### init and run ####
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('Initialized')
        ## summary ##
        merged = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter('logs/tarin', sess.graph)

        for step in xrange((NUM_EPOCHS * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:offset + BATCH_SIZE, ...]
            batch_labels = train_labels[offset:offset + BATCH_SIZE]

            feed_dict = {
                train_data_node: batch_data,
                train_label_node: batch_labels,
                keep_prob: KEEP_PROB,
            }
            sess.run(optimizer, feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:
                l, lr, predictions = sess.run(
                    [loss, learning_rate, train_prediction], feed_dict=feed_dict)
                pass_time = time.time() - start_time
                start_time = time.time()
                print('====== step %s run %.1f ======' % (step, pass_time))
                print('Minibatch loss %0.4f, learning_rate %0.6f' % (l, lr))
                print('Minibatch accuracy: %0.2f' % accuracy_rate(predictions, batch_labels))
                # cross validation
                cross_predictions = eval_in_batches(validation_data, sess)
                # cross_predictions = sess.run(train_eval, feed_dict={eval_data: validation_data})
                accuracy = accuracy_rate(cross_predictions, validation_labels)
                print('CrossValidation accuracy: %0.2f' % accuracy)
                print(cross_predictions.shape)
                print(validation_labels.shape)
                print(np.argmax(cross_predictions, axis=1))
                print(validation_labels)
                # print 8 error
                # condition = (np.argmax(cross_predictions, axis=1) != test_labels)
                # np.compress(condition, cross_predictions)

                # add summary
                tf.summary.scalar('Cross accuracy', accuracy)
                # summary_part = sess.run(merged, feed_dict={
                #             eval_data:test_data,
                #             train_data_node: batch_data,
                #             train_label_node: batch_labels,
                #             keep_prob: KEEP_PROB,
                #     })
                # writer_train.add_summary(summary_part, step)
                # accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fp16',
        default=False,
        help='use half float instead of full float',
        action='store_true'
    )
    parser.add_argument(
        '--self_test',
        default=False,
        help='True if running a self test',
        action='store_true'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    print([sys.argv[0]] + unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
