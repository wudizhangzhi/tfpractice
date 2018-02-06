# encoding: utf-8
from __future__ import print_function

import random
import time
import os

from captcha.image import ImageCaptcha
from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


'''
0.input parameter
1.load train data
2.split to validation data
3.create layer
4.compute loss
5.optimizer
6.init
7.run
8.add summary
9.predict
'''

config = {
    'data': '../X_train_color.npy',
    'label': '../y_train_color.npy',
    'NUM_EPOCHS': 2,
    'IMG_HEIGHT': 60,
    'IMG_WIDTH': 60,
    'CHANNEL': 3,
    'BATCH_SIZE': 100,
    'VALIDATION_SIZE': 100,
    'KEEP_PROB': 0.5,
    'NUM_LABEL': 10,
    'FREQUENCY': 50,  # 每隔多少部打印，执行
    'LR': 1e-4,  # learning rate
    'DECAY_RATE': 0.5,  # 衰减率
    'ONE_HOT': True,
    'LOG_DIR': 'logs/tarin',
    'SAVE_DIR': 'logs/model.ckpt-998',  # logs/model.ckpt
}


def plotrandom(images, labels, num=8, image_size=28, channel=1, one_hot=True):
    for i in range(num):
        index = np.random.randint(len(images))
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        if channel == 1:
            plt.imshow(images[index].reshape(image_size, image_size), cmap=plt.cm.gray_r, interpolation='nearest')
        else:
            plt.imshow(images[index].reshape(image_size, image_size, channel),
                       interpolation='nearest')
        # plt.title('Training: %s' % labels[index].tolist().index(1))
        if one_hot:
            plt.title('Prediction: %s' % np.argmax(labels[index, :]))
        else:
            plt.title('Prediction: %s' % np.argmax(labels[index]))
    # plt.ion()
    plt.show()


def plotimages(images, labels, image_size=28, channel=1, one_hot=True):
    num = len(images)
    for i in range(num):
        plt.subplot(2, num//2, i + 1)
        plt.axis('off')
        if channel == 1:
            plt.imshow(images[i].reshape(image_size, image_size), cmap=plt.cm.gray_r, interpolation='nearest')
        else:
            plt.imshow(images[i].reshape(image_size, image_size, channel),
                       interpolation='nearest')
        # plt.title('Training: %s' % labels[index].tolist().index(1))
        if one_hot:
            plt.title('Prediction: %s' % np.argmax(labels[i, :]))
        else:
            plt.title('Prediction: %s' % labels[i])
    plt.show()

class Captcha:
    def __init__(self, img_height=20, img_width=20, channel=3, num_epochs=1, batch_size=100, validation=64):
        self.NUM_EPOCHS = num_epochs  # 样本循环次数
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.CHANNEL = channel
        self.BATCH_SIZE = batch_size  # 每次训练的样本数量
        self.VALIDATION_SIZE = validation

    def readconfig(self, config):
        if isinstance(config, str):  # file path
            pass
        elif isinstance(config, dict):
            pass
        else:
            raise ValueError('config must be filepath or dict')
        print('==== reading config =====')
        self.FREQUENCY = config.get('FREQUENCY', 50)
        self.NUM_EPOCHS = config.get('NUM_EPOCHS', 10)
        self.NUM_LABEL = config.get('NUM_LABEL', 10)
        self.IMG_HEIGHT = config.get('IMG_HEIGHT', 40)
        self.IMG_WIDTH = config.get('IMG_WIDTH', 40)
        self.CHANNEL = config.get('CHANNEL', 3)
        self.BATCH_SIZE = config.get('BATCH_SIZE', 100)
        self.VALIDATION_SIZE = config.get('VALIDATION_SIZE', 64)
        self.KEEP_PROB = config.get('KEEP_PROB', 0.5)
        self.LR = config.get('LR', 0.1)
        self.DECAY_RATE = config.get('DECAY_RATE', 0.9)
        self.ONE_HOT = config.get('ONE_HOT', True)
        self.LOG_DIR = config.get('LOG_DIR', 'logs/')
        self.SAVE_DIR = config.get('SAVE_DIR', 'logs/model/')
        print('==== reading train data =====')
        self.DATA = np.load(config.get('data'))
        self.LABEL = np.load(config.get('label'))

        if tf.gfile.Exists(self.LOG_DIR):
            tf.gfile.DeleteRecursively(self.LOG_DIR)
        dirpath = os.path.dirname(self.LOG_DIR)
        print('create save path: %s' % dirpath)
        tf.gfile.MakeDirs(dirpath)
        # self.load_train_data(data, label)

    def model(self, data, is_train=False):
        print('==== create layers =====')
        data = data / 255.0
        # ==== conv1 ====
        with tf.name_scope('conv1'):
            data_conv1 = tf.nn.conv2d(data, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            relu_conv1 = tf.nn.relu(tf.nn.bias_add(data_conv1, self.bias_conv1))
            # relu_conv1 = tf.clip_by_value(relu_conv1, 1e-10, 1.)  # control data range
            pool_conv1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # ==== conv2 ====
        with tf.name_scope('conv2'):
            data_conv2 = tf.nn.conv2d(pool_conv1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            relu_conv2 = tf.nn.relu(tf.nn.bias_add(data_conv2, self.bias_conv2))
            # relu_conv2 = tf.clip_by_value(relu_conv2, 1e-10, 1.)  # control data range
            pool_conv2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # ==== fc1 =====
        with tf.name_scope('fc1'):
            pool_shape = pool_conv2.get_shape().as_list()
            pool_conv3_reshape = tf.reshape(pool_conv2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            data_fc1 = tf.nn.relu(tf.matmul(pool_conv3_reshape, self.W_fc1) + self.bias_fc1)
            # data_fc1 = tf.clip_by_value(data_fc1, 1e-10, 1.)  # control data range

        # ==== fc2 =====
        with tf.name_scope('fc2'):
            data_fc2 = tf.nn.relu(tf.matmul(data_fc1, self.W_fc2) + self.bias_fc2)
            # data_fc2 = tf.clip_by_value(data_fc2, 1e-10, 1.)  # control data range
        if is_train:
            data_fc2 = tf.nn.dropout(data_fc2, keep_prob=self.KEEP_PROB)
        # ==== fc3 =====
        with tf.name_scope('fc3'):
            logits = tf.matmul(data_fc2, self.W_fc3) + self.bias_fc3
        return logits

    def compute_accuracy(self, prediction, label):
        # ont hot
        if self.ONE_HOT:
            # return 100.0 * np.sum(np.equal(np.argmax(prediction, axis=1), np.argmax(label, axis=1)),
            #                       dtype='float32') / (prediction.shape[0])
            return 100.0 * tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(prediction, axis=1), tf.argmax(label, axis=1)), tf.float32))
        else:
            # return 100.0 * np.sum(np.equal(np.argmax(prediction, axis=1), label), dtype='float32') / (
            #     prediction.shape[0])
            correct = tf.equal(tf.argmax(prediction, axis=1), label)
            return 100.0 * tf.reduce_mean(tf.cast(correct, tf.float32))

    def weight_variable(self, shape, name=''):
        stddev = np.sqrt(2 / np.prod(shape[1:]))
        return tf.Variable(
            tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=stddev), name=name)

    def define_variables(self):
        # ==== conv1 ====
        IMAGE_PIXELS = self.IMG_WIDTH * self.IMG_HEIGHT * self.CHANNEL
        self.W_conv1 = tf.Variable(
            tf.truncated_normal(shape=[5, 5, self.CHANNEL, 32], dtype=tf.float32,
                                stddev=1.0 / np.sqrt(float(IMAGE_PIXELS))), name='W_conv1')
        # self.bias_conv1 = tf.Variable(tf.zeros(shape=[32], dtype=tf.float32), name='bias_conv1')
        self.bias_conv1 = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32), name='bias_conv1')
        # ==== conv2 ====
        self.W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], dtype=tf.float32,
                                                       stddev=0.1),
                                   name='W_conv2')
        # self.bias_conv2 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32), name='bias_conv2')
        self.bias_conv2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias_conv2')
        # self.bias_conv2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='bias_conv2')

        # ==== fc1 ====
        self.W_fc1 = tf.Variable(
            tf.truncated_normal(shape=[self.IMG_HEIGHT // 4 * self.IMG_WIDTH // 4 * 64, 1024], dtype=tf.float32),
            name='W_fc1')
        # self.bias_fc1 = tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32), name='bias_fc1')
        self.bias_fc1 = tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32), name='bias_fc1')
        # self.bias_fc1 = tf.Variable(tf.constant(0.001, shape=[1024], dtype=tf.float32), name='bias_fc1')
        # ==== fc2 ====
        self.W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 1024], dtype=tf.float32), name='W_fc2')
        # self.bias_fc2 = tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32), name='bias_fc2')
        self.bias_fc2 = tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32), name='bias_fc2')
        # self.bias_fc2 = tf.Variable(tf.constant(0.001, shape=[1024], dtype=tf.float32), name='bias_fc2')
        # ==== fc3 ====
        self.W_fc3 = tf.Variable(tf.truncated_normal(shape=[1024, self.NUM_LABEL], dtype=tf.float32), name='W_fc3')
        self.bias_fc3 = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABEL], dtype=tf.float32), name='bias_fc2')
        # self.bias_fc3 = tf.Variable(tf.constant(0.001, shape=[self.NUM_LABEL], dtype=tf.float32), name='bias_fc3')

    def conv2d_layer(self, input, in_channel, out_channel, name='', reuse=False, wd=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            # with tf.name_scope(name):
            stddev = 1.0 / tf.sqrt(tf.cast(tf.reduce_prod(input.shape[1:]), tf.float32))
            # stddev = 0.1
            initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
            W = tf.get_variable('W',
                                shape=[5, 5, in_channel, out_channel],
                                initializer=initializer,
                                )
            initializer = tf.constant_initializer(0.1)
            # initializer = tf.zeros_initializer()
            bias = tf.get_variable('bias',
                                   shape=[out_channel],
                                   dtype=tf.float32,
                                   initializer=initializer
                                   )
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            relu = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
            pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

            if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(pool), wd, name='weight_loss')
                bias_decay = tf.multiply(tf.nn.l2_loss(W), wd, name='bias_loss')
                tf.add_to_collection('losses', weight_decay)
                tf.add_to_collection('losses', bias_decay)

            tf.summary.histogram('W', W)
            tf.summary.histogram('bias', bias)
            tf.summary.histogram('conv', conv)
            tf.summary.histogram('relu', relu)
            tf.summary.histogram('pool', pool)
        return pool

    def fc_layer(self, input, in_channel, out_channel, name='', relu=True, reuse=False, wd=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            # with tf.name_scope(name):
            stddev = 1.0 / tf.sqrt(tf.cast(tf.reduce_prod(input.shape[1:]), tf.float32))
            # stddev = 0.1
            initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
            W = tf.get_variable('W',
                                shape=[in_channel, out_channel],
                                initializer=initializer
                                )
            initializer = tf.constant_initializer(0.1)
            # initializer = tf.zeros_initializer()
            bias = tf.get_variable('bias',
                                   shape=[out_channel],
                                   dtype=tf.float32,
                                   initializer=initializer
                                   )
            logits = tf.nn.bias_add(tf.matmul(input, W), bias, name='logits')
            if relu:
                logits = tf.nn.relu(logits, name='fc')

            if wd is not None:
                weight_decay = tf.multiply(tf.nn.l2_loss(W), wd, name='weight_loss')
                bias_decay = tf.multiply(tf.nn.l2_loss(W), wd, name='bias_loss')
                tf.add_to_collection('losses', weight_decay)
                tf.add_to_collection('losses', bias_decay)

            tf.summary.histogram('W', W)
            tf.summary.histogram('bias', bias)
            tf.summary.histogram('linear', logits)
        return logits

    def inference(self, images, is_train=True, reuse=False):
        # reshape images
        # ==== conv1 ====
        data_conv1 = self.conv2d_layer(images, in_channel=self.CHANNEL, out_channel=32, name='conv1', reuse=reuse)
        # ==== conv2 ====
        data_conv2 = self.conv2d_layer(data_conv1, in_channel=32, out_channel=64, name='conv2', reuse=reuse)
        # ==== fc1 ====
        # reshape
        out_channel = self.IMG_HEIGHT // 4 * self.IMG_WIDTH // 4 * 64
        data_conv2 = tf.reshape(data_conv2, [-1, out_channel])
        data_fc1 = self.fc_layer(data_conv2, in_channel=out_channel, out_channel=512, name='fc1', reuse=reuse, wd=0.004)
        # ==== fc2 ====
        data_fc2 = self.fc_layer(data_fc1, in_channel=512, out_channel=1024, name='fc2', reuse=reuse, wd=0.004)
        # ==== fc3 =====
        logits = self.fc_layer(data_fc2, in_channel=1024, out_channel=self.NUM_LABEL, name='fc3', relu=False,
                               reuse=reuse)
        if is_train:
            logits = tf.nn.dropout(logits, keep_prob=self.KEEP_PROB)
        return logits

    def loss(self, logits, labels):
        with tf.name_scope('compute_loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.add_to_collection('losses', loss)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.summary.scalar('loss', total_loss)
            tf.summary.histogram('cross_entropy', cross_entropy)
        return total_loss

    def main(self):
        # shuffle data
        data_size = self.DATA.shape[0]
        sample_index = np.random.choice(data_size, data_size)
        data_shuffle = self.DATA[sample_index, :]
        if self.ONE_HOT:
            label_shuffle = self.LABEL[sample_index, :]
        else:
            label_shuffle = self.LABEL[sample_index]

        validation_data = data_shuffle[:self.VALIDATION_SIZE, :]
        if self.ONE_HOT:
            validation_label = label_shuffle[:self.VALIDATION_SIZE, :]
        else:
            validation_label = label_shuffle[:self.VALIDATION_SIZE]

        train_data = data_shuffle[self.VALIDATION_SIZE:, :]
        if self.ONE_HOT:
            train_label = label_shuffle[self.VALIDATION_SIZE:, :]
        else:
            train_label = label_shuffle[self.VALIDATION_SIZE:]
        train_size = train_data.shape[0]

        # placeholder
        print('==== define placeholders =====')
        train_data_node = tf.placeholder(shape=[self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL],
                                         dtype=tf.float32, name='train_data_node')
        if self.ONE_HOT:
            train_label_node = tf.placeholder(shape=[self.BATCH_SIZE, self.NUM_LABEL], dtype=tf.float32,
                                              name='train_label_node')
        else:
            train_label_node = tf.placeholder(shape=[self.BATCH_SIZE], dtype=tf.float32,
                                              name='train_label_node')
        # keep_prob = tf.constant(dtype=tf.float32)
        eval_cross_data = tf.placeholder(
            shape=[self.VALIDATION_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL], dtype=tf.float32,
            name='eval_cross_data')
        if self.ONE_HOT:
            eval_cross_label = tf.placeholder(
                shape=[self.VALIDATION_SIZE, self.NUM_LABEL], dtype=tf.float32,
                name='eval_cross_data')
        else:
            eval_cross_label = tf.placeholder(
                shape=[self.VALIDATION_SIZE], dtype=tf.float32,
                name='eval_cross_data')

        logits = self.inference(train_data_node)

        loss = self.loss(logits, train_label_node)

        with tf.name_scope('predict'):
            train_prediction = tf.nn.softmax(logits)
        cross_prediction = tf.nn.softmax(self.inference(eval_cross_data, is_train=False, reuse=True))

        with tf.name_scope('optimizer'):
            batch = tf.Variable(0, dtype=tf.float32)
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
                self.LR,  # Base learning rate.
                batch * self.BATCH_SIZE,  # Current index into the dataset.
                train_size,  # Decay step.
                self.DECAY_RATE,  # Decay rate.
                staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

        with tf.name_scope('accuracy'):
            accuracy_train = self.compute_accuracy(train_prediction, train_label_node)
            tf.summary.scalar('accuracy_train', accuracy_train)
            accuracy_cross = self.compute_accuracy(cross_prediction, eval_cross_label)
            tf.summary.scalar('accuracy_cross', accuracy_cross)

        with tf.Session() as sess:
            ## summary ##
            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            writer_train = tf.summary.FileWriter(self.LOG_DIR, sess.graph)
            sess.run(tf.global_variables_initializer())

            for step in xrange((self.NUM_EPOCHS * train_size) // self.BATCH_SIZE):
                num_epochs = (step * self.BATCH_SIZE) // train_size
                offset = (step * train_size) % (train_size - self.BATCH_SIZE)

                batch_data = train_data[offset:offset + self.BATCH_SIZE, :]
                batch_data = batch_data.reshape([self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL])
                if self.ONE_HOT:
                    batch_label = train_label[offset:offset + self.BATCH_SIZE, :]
                else:
                    batch_label = train_label[offset:offset + self.BATCH_SIZE]
                feed_dict = {
                    train_data_node: batch_data,
                    train_label_node: batch_label,
                }

                sess.run(optimizer, feed_dict=feed_dict)
                start_time = time.time()
                if step % self.FREQUENCY == 0:
                    pass_time = time.time() - start_time
                    start_time = time.time()
                    print('===== step %s  epoch %s  time %s  ====' % (step, num_epochs, pass_time))
                    batch_loss, batch_prediction, lr = sess.run([loss, train_prediction, learning_rate],
                                                                feed_dict=feed_dict)
                    print('batch loss: %0.4f  leaning rate : %g' % (batch_loss, lr))
                    accuracy = sess.run(accuracy_train, feed_dict=feed_dict)
                    print('batch accuracy: %0.2f' % accuracy)
                    feed_dict_cross = {
                        eval_cross_data: validation_data.reshape(
                            [self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL]),
                        eval_cross_label: validation_label
                    }
                    accuracy = sess.run(accuracy_cross, feed_dict=feed_dict_cross)
                    print('cross accuracy: %0.2f' % accuracy)
                    feed_dict = {
                        eval_cross_data: validation_data.reshape(
                            [self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL]),
                        eval_cross_label: validation_label,
                        train_data_node: batch_data,
                        train_label_node: batch_label,
                    }
                    summary_part = sess.run(merged, feed_dict=feed_dict)
                    writer_train.add_summary(summary_part, step)

            # tf.gfile.MakeDirs(self.SAVE_DIR)
            saver.save(sess, self.SAVE_DIR, global_step=batch)

    def train(self):
        with tf.Session() as sess:
            # shuffle data
            data_size = self.DATA.shape[0]
            sample_index = np.random.choice(data_size, data_size)
            data_shuffle = self.DATA[sample_index, :]
            if self.ONE_HOT:
                label_shuffle = self.LABEL[sample_index, :]
            else:
                label_shuffle = self.LABEL[sample_index]

            validation_data = data_shuffle[:self.VALIDATION_SIZE, :]
            if self.ONE_HOT:
                validation_label = label_shuffle[:self.VALIDATION_SIZE, :]
            else:
                validation_label = label_shuffle[:self.VALIDATION_SIZE]

            train_data = data_shuffle[self.VALIDATION_SIZE:, :]
            if self.ONE_HOT:
                train_label = label_shuffle[self.VALIDATION_SIZE:, :]
            else:
                train_label = label_shuffle[self.VALIDATION_SIZE:]
            train_size = train_data.shape[0]

            ###### variabels ######
            print('==== define varibales placeholders =====')
            train_data_node = tf.placeholder(shape=[self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL],
                                             dtype=tf.float32, name='train_data_node')
            if self.ONE_HOT:
                train_label_node = tf.placeholder(shape=[self.BATCH_SIZE, self.NUM_LABEL], dtype=tf.float32,
                                                  name='train_label_node')
            else:
                train_label_node = tf.placeholder(shape=[self.BATCH_SIZE], dtype=tf.float32,
                                                  name='train_label_node')
            # keep_prob = tf.constant(dtype=tf.float32)
            eval_cross_data = tf.placeholder(
                shape=[self.VALIDATION_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL], dtype=tf.float32,
                name='eval_cross_data')
            if self.ONE_HOT:
                eval_cross_label = tf.placeholder(
                    shape=[self.VALIDATION_SIZE, self.NUM_LABEL], dtype=tf.float32,
                    name='eval_cross_data')
            else:
                eval_cross_label = tf.placeholder(
                    shape=[self.VALIDATION_SIZE], dtype=tf.float32,
                    name='eval_cross_data')

            self.define_variables()

            logits = self.model(train_data_node, is_train=True)

            # loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(labels=train_label_node, logits=logits))
            # train_label_node = tf.cast(train_label_node, tf.int64)
            # eval_cross_label = tf.cast(eval_cross_label, tf.int64)

            if self.ONE_HOT:
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=train_label_node, logits=logits))
            else:
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label_node, logits=logits))
            # L2 regularization for the fully connected parameters.
            # regularizers = (tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.bias_fc1) +
            #                 tf.nn.l2_loss(self.W_fc2) + tf.nn.l2_loss(self.bias_fc2))
            # # Add the regularization term to the loss.
            # loss += 5e-4 * regularizers
            tf.summary.scalar('loss', loss)

            batch = tf.Variable(0, dtype=tf.float32)
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
                self.LR,  # Base learning rate.
                batch * self.BATCH_SIZE,  # Current index into the dataset.
                train_size,  # Decay step.
                self.DECAY_RATE,  # Decay rate.
                staircase=True)

            # optimizer
            # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
            train_prediction = tf.nn.softmax(logits)

            accuracy_train = self.compute_accuracy(train_prediction, train_label_node)
            tf.summary.scalar('accuracy_train', accuracy_train)

            cross_prediction = tf.nn.softmax(self.model(eval_cross_data, is_train=False))
            accuracy_cross = self.compute_accuracy(cross_prediction, eval_cross_label)
            tf.summary.scalar('accuracy_cross', accuracy_cross)

            # init
            print('==== init start training =====')
            ## summary ##
            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            writer_train = tf.summary.FileWriter(self.LOG_DIR, sess.graph)
            sess.run(tf.global_variables_initializer())

            for step in xrange((self.NUM_EPOCHS * train_size) // self.BATCH_SIZE):
                num_epochs = (step * self.BATCH_SIZE) // train_size
                offset = (step * train_size) % (train_size - self.BATCH_SIZE)

                batch_data = train_data[offset:offset + self.BATCH_SIZE, :]
                batch_data = batch_data.reshape([self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL])
                if self.ONE_HOT:
                    batch_label = train_label[offset:offset + self.BATCH_SIZE, :]
                else:
                    batch_label = train_label[offset:offset + self.BATCH_SIZE]
                feed_dict = {
                    train_data_node: batch_data,
                    train_label_node: batch_label,
                }

                sess.run(optimizer, feed_dict=feed_dict)
                start_time = time.time()
                if step % self.FREQUENCY == 0:
                    pass_time = time.time() - start_time
                    start_time = time.time()
                    print('===== step %s  epoch %s  time %s  ====' % (step, num_epochs, pass_time))
                    batch_loss, batch_prediction, lr = sess.run([loss, train_prediction, learning_rate],
                                                                feed_dict=feed_dict)
                    print('batch loss: %0.4f  leaning rate : %g' % (batch_loss, lr))
                    accuracy = sess.run(accuracy_train, feed_dict=feed_dict)
                    print('batch accuracy: %0.2f' % accuracy)
                    feed_dict_cross = {
                        eval_cross_data: validation_data.reshape(
                            [self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL]),
                        eval_cross_label: validation_label
                    }
                    accuracy = sess.run(accuracy_cross, feed_dict=feed_dict_cross)
                    print('cross accuracy: %0.2f' % accuracy)
                    feed_dict = {
                        eval_cross_data: validation_data.reshape(
                            [self.BATCH_SIZE, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL]),
                        eval_cross_label: validation_label,
                        train_data_node: batch_data,
                        train_label_node: batch_label,
                    }
                    summary_part = sess.run(merged, feed_dict=feed_dict)
                    writer_train.add_summary(summary_part, step)

                    # if accuracy_cross > 30:
                    #     plotrandom(batch_data, batch_prediction, num=8, image_size=self.IMG_HEIGHT, channel=3)

            saver.save(sess, self.SAVE_DIR, global_step=batch)
            # writer_train.flush()

    def test_mnist(self):
        print('==== load mnist test data  ====')
        from getstarted import input_data
        data_sets = input_data.read_data_sets('getstarted/data')
        self.DATA = data_sets.train.images
        self.LABEL = data_sets.train.labels
        self.IMG_HEIGHT = 28
        self.IMG_WIDTH = 28
        self.CHANNEL = 1
        self.ONE_HOT = False
        self.train()

    def delete_old_log(self):
        print('==== delete old log ====')
        if tf.gfile.Exists(self.LOG_DIR):
            tf.gfile.DeleteRecursively(self.LOG_DIR)
        tf.gfile.MakeDirs(self.LOG_DIR)

    def predict(self, images):

        with tf.Session() as sess:
            shape = [-1, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL]
            images = images.reshape(shape)
            input_data = tf.placeholder(shape=images.shape, dtype=tf.float32)
            prediction = tf.nn.softmax(self.inference(input_data, is_train=False))

            prediction_label = tf.argmax(prediction, axis=1)
            saver = tf.train.Saver()
            saver.restore(sess, self.SAVE_DIR)

            prediction_label_index = sess.run(prediction_label, feed_dict={input_data: images})

        return prediction_label_index

    def predict_from_dir(self, dirname):
        filelist = os.listdir(dirname)
        img_array = None
        image_shape = None
        for filename in filelist:
            filepath = os.path.join(dirname, filename)
            img = np.array(Image.open(filepath), dtype='float32')
            image_shape = img.shape
            img = img.reshape([1, np.prod(image_shape)])
            if img_array is None:
                img_array = img
            else:
                img_array = np.vstack((img_array, img))
        # img_array = img_array.reshape([-1, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL])
        predictions = self.predict(img_array)
        plotimages(img_array.reshape([-1, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL]), predictions,
                   image_size=image_shape[1], channel=3, one_hot=False)

    def test_model_form_file(self, test_X, test_label):
        X_test = np.load(test_X)
        label_test = np.load(test_label)
        test_size = len(label_test)
        X_test = X_test.reshape([-1, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL])

        input_data = tf.placeholder(shape=X_test.shape, dtype=tf.float32)
        label_batch = tf.placeholder(shape=label_test.shape, dtype=tf.float32)
        with tf.Session() as sess:
            prediction = tf.nn.softmax(self.inference(input_data, is_train=False))
            accuracy = self.compute_accuracy(prediction, label_batch)
            saver = tf.train.Saver()
            saver.restore(sess, self.SAVE_DIR)
            test_accuracy = sess.run(accuracy, feed_dict={input_data: X_test, label_batch: label_test})
            print('===== accuracy: %0.2f =====' % test_accuracy)

    def predict_one_image_from_file(self, image):
        input = np.loads(image)

        X_input = input.reshape([1, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL])

        input_data = tf.placeholder(shape=X_input.shape, dtype=tf.float32)
        with tf.Session() as sess:
            prediction = tf.nn.softmax(self.inference(input_data, is_train=False))
            saver = tf.train.Saver()
            saver.restore(sess, self.SAVE_DIR)
            x_prediction = sess.run(prediction)

        plt.imshow(input)

    def predict_one_image(self, image, label=None):
        X_input = image.reshape([1, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL])
        input_data = tf.placeholder(shape=X_input.shape, dtype=tf.float32)
        with tf.Session() as sess:
            prediction = tf.nn.softmax(self.inference(input_data, is_train=False))
            saver = tf.train.Saver()
            saver.restore(sess, self.SAVE_DIR)
            x_prediction = sess.run(prediction, feed_dict={input_data: X_input})

        plt.imshow(image)
        plt.title('Prediction: %s' % np.argmax(x_prediction))
        plt.show()


class OCRIter(object):
    def __init__(self, count=4, height=60, width=60):
        super(OCRIter, self).__init__()
        self.letters = '0123456789'
        self.count = count
        self.height = height
        self.width = width
        self.captcha = ImageCaptcha(fonts=['../arial.ttf'], width=width * count, height=height)

    def __next__(self):
        chars = ''
        for _ in xrange(self.count):
            chars += random.choice(self.letters)
        self.captcha.write(chars, 'tmp.png')
        img = np.array(Image.open('tmp.png'), dtype='float32')
        return img, chars


if __name__ == '__main__':
    orciter = OCRIter(count=1)
    img, chars = next(orciter)

    captcha = Captcha()
    captcha.readconfig(config)
    # captcha.main()
    # captcha.predict_from_dir('../data_test')
    # captcha.test_model_form_file(test_X='../X_test_color.npy', test_label='../y_test_color.npy')
    # captcha.predict_one_image_from_file('../')


    captcha.predict_one_image(img)
