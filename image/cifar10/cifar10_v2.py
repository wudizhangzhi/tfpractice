import datetime

import numpy as np
import tensorflow as tf
import os

import time

import cifar10_input

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
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
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
        pool2_flatten = tf.reshape(pool2, (pool2.get_shape().as_list()[0], -1))
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
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    class _LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._step = -1
            self._start_time = int(time.time())

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)

        def after_run(self,
                      run_context,  # pylint: disable=unused-argument
                      run_values):
            if self._step % 100 == 0:
                current_time = int(time.time())
                duration = current_time - self._start_time
                self._start_time = current_time

                _loss = run_values.results
                example_per_sec = float(100 / duration)

                format_str = '%s: step: %d, loss: %0.2f (%0.1f example/sec)'

                print(format_str % (datetime.datetime.now(),
                                    self._step,
                                    _loss,
                                    example_per_sec
                                    ))

    init_op = tf.global_variables_initializer()
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir='save',
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.train_step),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
        mon_sess.run(init_op)
        while not mon_sess.should_stop():
            mon_sess.run(train_op)


def main(_):
    if FLAGS.is_train:
        global_step = tf.train.get_or_create_global_step()
        images, labels = cifar10_input.distorted_inputs('cifar-10-batches-bin', 100)
        logits = inference(images)
        _loss = loss(logits, labels)
        train(_loss, global_step)


if __name__ == '__main__':
    tf.app.run()
