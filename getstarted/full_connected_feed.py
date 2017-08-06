# encoding: utf-8

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import division
from __future__ import print_function

import argparse
import time
import os
import sys
from six.moves import xrange

import tensorflow as tf
import numpy as np
import input_data
import mnist

FLAGS = None


def placeholder_inputs(batchsize, image_pixels):
    image_placeholder = tf.placeholder(shape=[batchsize, image_pixels], dtype=tf.float32)
    labels_placeholder = tf.placeholder(shape=[batchsize], dtype=tf.int32)
    return image_placeholder, labels_placeholder


def fill_feed_dict(data_sets, image_placeholder, labels_placeholder):
    image_feed, labels_feed = data_sets.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    return {
        image_placeholder: image_feed,
        labels_placeholder: labels_feed,
    }


def do_eval(sess, eval_correct, image_placeholder, labels_placeholder, data_set):
    true_count = 0
    step_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = step_per_epoch * FLAGS.batch_size
    for step in xrange(step_per_epoch):
        feed_dict = fill_feed_dict(data_set, image_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        image_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size, mnist.IMAGE_PIEXLS)

        logits = mnist.inference(image_placeholder, FLAGS.hidden_unt1, FLAGS.hidden_unt2)

        loss = mnist.loss(logits, labels_placeholder)

        train_op = mnist.training(loss, learning_rate=FLAGS.lr)

        eval_correct = mnist.evalution(logits, labels_placeholder)

        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            # Run the Op to initialize the variables.
            sess.run(init)

            feed_dict = fill_feed_dict(data_sets.train,
                                       image_placeholder,
                                       labels_placeholder)

            for step in xrange(FLAGS.max_steps):
                start_time = time.time()

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time

                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            image_placeholder,
                            labels_placeholder,
                            data_sets.train)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            image_placeholder,
                            labels_placeholder,
                            data_sets.validation)
                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            image_placeholder,
                            labels_placeholder,
                            data_sets.test)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    # parser parameters
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/',
        help='Direction to put log data'
    )

    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='data/',
        help='Direction to save train data'
    )

    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='number of steps to run trainer'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='number of data size in one step'
    )

    parser.add_argument(
        '--hidden_unt1',
        type=int,
        default=128,
        help='number of hidden 1 unit'
    )

    parser.add_argument(
        '--hidden_unt2',
        type=int,
        default=32,
        help='number of hidden 2 unit'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate '
    )

    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true',
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
