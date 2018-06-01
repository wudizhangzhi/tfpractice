import numpy as np
import tensorflow as tf
import os
import generate_data

flags = tf.app.flags
# string
flags.DEFINE_string('datapath', 'data', '数据路径')
# integer
flags.DEFINE_integer('epoch_num', 10, '训练周期')
flags.DEFINE_integer('train_step', 5000, '训练周期')
flags.DEFINE_integer('width', 32, '宽')
flags.DEFINE_integer('height', 32, '高')
flags.DEFINE_integer('channel', 3, 'channels')
flags.DEFINE_integer('classes', 10, '总类别')
flags.DEFINE_integer('batch_size', 128, '训练样本大小')
flags.DEFINE_integer('predict_num', 10, '预测数量')
# float
flags.DEFINE_float('lr', 0.001, '学习率')
flags.DEFINE_float('lr_decay', 0.9, '学习率衰退率')
flags.DEFINE_float('lr_decay_step', 5000, '学习率衰退率')
flags.DEFINE_float('keep_prob', 0.75, '保留率')
# boolean
flags.DEFINE_boolean('is_train', True, '是否是训练')
flags.DEFINE_boolean('is_plt', False, '是否显示图表')
FLAGS = flags.FLAGS


def inference(images):
    # reshape ?

    # conv1
    with tf.name_scope('Conv1'):
        conv1 = tf.layers.conv2d(
            images,
            filters=64,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            bias_initializer=tf.zeros_initializer(),
            activation=tf.nn.relu,
        )

        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=[3, 3],
            strides=[2, 2],
            padding='same'
        )

        # TODO lrn normal ?
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

    # conv2
    with tf.name_scope('Conv2'):
        conv2 = tf.layers.conv2d(
            norm1,
            filters=64,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            bias_initializer=tf.constant_initializer(0.1),
            activation=tf.nn.relu,
        )

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')

        pool2 = tf.layers.max_pooling2d(
            norm2,
            pool_size=[3, 3],
            strides=[2, 2],
            padding='same'
        )

    # local3
    with tf.name_scope('Local3'):
        pool2_flatten = tf.reshape(pool2, (pool2.shape().get_list()[0], -1))
        local3 = tf.layers.dense(
            pool2_flatten,
            units=384,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.04),
            bias_initializer=tf.constant_initializer(0.1)
        )

    # local4
    with tf.name_scope('Local4'):
        local4 = tf.layers.dense(
            local3,
            units=192,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.04),
            bias_initializer=tf.constant_initializer(0.1)
        )

    # output
    with tf.name_scope('Output'):
        output = tf.layers.dense(
            local4,
            units=192,
            kernel_initializer=tf.truncated_normal_initializer(stddev=1 / 192.0),
            bias_initializer=tf.constant_initializer(0.0)
        )
    return output


def loss(logits, labels):
    lables = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=lables,
        name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


def train(loss, global_step):
    with tf.name_scope('Train'):
        tf.train.AdamOptimizer().minimize(loss)
