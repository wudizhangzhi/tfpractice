#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/27 下午3:34
# @Author  : wudizhangzhi


import tensorflow as tf
import mobilenet_v1
import sys

# TODO
sys.path.append('/Users/zhangzhichao/github/tfpractice/image')
from cifar10 import cifar10_input

slim = tf.contrib.slim

flags = tf.app.flags
# int
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('num_classes', 10, 'Number of classes')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100,
                     'How often to save checkpoints, secs')
# string
flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('dataset_dir', '../data',
                    'dataset dir')
flags.DEFINE_string('checkpoint_dir', 'log',
                    'Directory for writing training checkpoints and logs')
# float
flags.DEFINE_float('depth_multiplier', 1, '')

# boolean
flags.DEFINE_bool('quantize', False, 'Quantize training')

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94


def imagenet_input(is_training):
    """
    返回数据
    :param is_training: 
    """
    print('load data')
    import sys
    sys.path.append('..')
    from datasets import dataset_factory
    if is_training:
        dataset = dataset_factory.get_dataset('cifar10', 'train', FLAGS.dataset_dir)
    else:
        dataset = dataset_factory.get_dataset('cifar10', 'validation', FLAGS.dataset_dir)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=is_training,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size
    )
    [image, label] = provider.get(['image', 'label'])

    # TODO
    # image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    #     'mobilenet_v1', is_training=is_training)
    #
    # image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=4,
        capacity=5 * FLAGS.batch_size)
    labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
    return images, labels


def get_quant_delay():
    if FLAGS.fine_tune_checkpoint:
        return 0
    else:
        return 2500


def get_learning_rate():
    if FLAGS.fine_tune_checkpoint:
        # If we are fine tuning a checkpoint we need to start at a lower learning
        # rate since we are farther along on training.
        return 1e-4
    else:
        return 0.045


def build_model():
    print('build model')
    g = tf.Graph()
    with g.as_default(), tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        # inputs, labels = imagenet_input(is_training=True)
        inputs, labels = cifar10_input.distorted_inputs('../cifar10/cifar-10-batches-bin', 100)
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
            print('default layers')
            logits, _ = mobilenet_v1.mobilenet_v1(
                inputs,
                is_training=True,
                depth_multiplier=FLAGS.depth_multiplier,
                num_classes=FLAGS.num_classes,
            )

        print('start train')
        one_hot_labels = tf.one_hot(tf.cast(labels, dtype=tf.uint8), FLAGS.num_classes, dtype=tf.float32)
        slim.losses.softmax_cross_entropy(one_hot_labels, logits)
        if FLAGS.quantize:
            tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())

        total_loss = tf.losses.get_total_loss(name='total_loss')
        # configure the learning rate using an exponential decay
        num_epochs_per_decay = 2.5
        imagenet_size = 1271167
        decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)

        learning_rate = tf.train.exponential_decay(
            get_learning_rate(),
            tf.train.get_or_create_global_step(),
            decay_steps,
            _LEARNING_RATE_DECAY_FACTOR,
            staircase=True
        )

        opt = tf.train.GradientDescentOptimizer(learning_rate)

        train_tensor = slim.learning.create_train_op(
            total_loss,
            optimizer=opt
        )

    slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
    slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'traing')
    return g, train_tensor


def get_checkpoint_init_fn():
    """Returns the checkpoint init_fn if the checkpoint is provided."""
    if FLAGS.fine_tune_checkpoint:
        variables_to_restore = slim.get_variables_to_restore()
        global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
        slim_init_fn = slim.assign_from_checkpoint_fn(
            FLAGS.fine_tune_checkpoint,
            variables_to_restore,
            ignore_missing_vars=True
        )

        def init_fn(sess):
            slim_init_fn(sess)
            sess.run(global_step_reset)

        return init_fn
    else:
        return None


def debug():
    g = tf.Graph()
    inputs = tf.placeholder(shape=[None, 28, 28, 3], dtype=tf.float32)
    labels = tf.placeholder(shape=[None, ], dtype=tf.float32)
    with g.as_default(), tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        # inputs, labels = imagenet_input(is_training=True)
        tf_inputs, tf_labels = cifar10_input.distorted_inputs('../cifar10/cifar-10-batches-bin', 100)
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
            logits, _ = mobilenet_v1.mobilenet_v1(
                inputs,
                is_training=True,
                depth_multiplier=FLAGS.depth_multiplier,
                num_classes=FLAGS.num_classes,
            )

        one_hot_labels = tf.one_hot(tf.cast(labels, dtype=tf.uint8), FLAGS.num_classes, dtype=tf.float32)
        slim.losses.softmax_cross_entropy(one_hot_labels, logits)
        if FLAGS.quantize:
            tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())

        total_loss = tf.losses.get_total_loss(name='total_loss')
        # configure the learning rate using an exponential decay
        num_epochs_per_decay = 2.5
        imagenet_size = 1271167
        decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)

        learning_rate = tf.train.exponential_decay(
            get_learning_rate(),
            tf.train.get_or_create_global_step(),
            decay_steps,
            _LEARNING_RATE_DECAY_FACTOR,
            staircase=True
        )

        opt = tf.train.GradientDescentOptimizer(learning_rate)

        train_tensor = slim.learning.create_train_op(
            total_loss,
            optimizer=opt
        )

    slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
    slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'traing')

    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    # accuracy
    test_inputs, test_labels = cifar10_input.distorted_inputs('../cifar10/cifar-10-batches-bin', 1000,
                                                              is_validdata=True)
    accuracy = tf.metrics.accuracy(tf.argmax(logits, axis=1), test_labels)

    with tf.Session(graph=g) as session:
        session.run(init_op)

        for step in range(10000):
            _loss, _ = session.run([total_loss, train_tensor], feed_dict={
                inputs:tf_inputs,
                labels: tf_labels,
            })
            if step % 100 == 0:
                _accuracy = session.run(accuracy, feed_dict={
                    inputs: test_inputs,
                    labels: test_labels
                })
                print('step: %d, loss: %s, accuracy: %s' % (step, _loss, _accuracy))


def main(_):
    debug()
    # g, train_tensor = build_model()
    #
    # with g.as_default():
    #     slim.learning.train(
    #         train_tensor,
    #         FLAGS.checkpoint_dir,
    #         is_chief=(FLAGS.task == 0),
    #         master=FLAGS.master,
    #         log_every_n_steps=FLAGS.log_every_n_steps,
    #         graph=g,
    #         number_of_steps=FLAGS.number_of_steps,
    #         save_summaries_secs=FLAGS.save_summaries_secs,
    #         save_interval_secs=FLAGS.save_interval_secs,
    #         init_fn=get_checkpoint_init_fn(),
    #         global_step=tf.train.get_global_step()
    #     )


if __name__ == '__main__':
    tf.app.run()
