# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import re
import tensorflow.contrib.slim as slim
from network_base import mobilenet_v1, mobilenet_v1_base
from read_data import produce_test_data, read_label_name, produce_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log',
                           """Directory where to save log. """)
tf.app.flags.DEFINE_string('data_dir', 'cifar-10-batches-py',
                           """Directory where to load data. """)
tf.app.flags.DEFINE_string('traindata_prefix', 'data_batch_*',
                           """train data file name pattern""")
tf.app.flags.DEFINE_string('testdata_name', 'test_batch',
                           """test data file name""")
tf.app.flags.DEFINE_string('namespace_file', 'batches.meta',
                           """label name list""")
tf.app.flags.DEFINE_integer('n_labels', 10,
                            """Num of lables""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch size""")
tf.app.flags.DEFINE_integer('epochs', 20,
                            """epochs""")
tf.app.flags.DEFINE_integer('maxstep', 100000,
                            """maxstep""")
tf.app.flags.DEFINE_float('lr', 0.001,
                          """learning rate""")
tf.app.flags.DEFINE_float('lr_decay_rate', 0.9,
                          """lr decay rate""")
tf.app.flags.DEFINE_integer('decay_per_step', 10000,
                            """num of step per decay""")
tf.app.flags.DEFINE_integer('width', 32,
                            """image width""")
tf.app.flags.DEFINE_integer('height', 32,
                            """image height""")
tf.app.flags.DEFINE_integer('channel', 3,
                            """image channel""")


# tf.app.flags.DEFINE_boolean('is_save', False,
#                             """whether save params""")


def main(argv):
    # read data
    print('=== read data ===')
    label_names = read_label_name(os.path.join(FLAGS.data_dir, FLAGS.namespace_file))
    # 获取训练数据

    filelist = os.listdir(FLAGS.data_dir)
    train_filelist = []
    for f in filelist:
        if re.match(FLAGS.traindata_prefix, f):
            train_filelist.append(os.path.join(FLAGS.data_dir, f))

    train_images, train_lables = produce_data(train_filelist)
    test_images, test_labels = produce_test_data(os.path.join(FLAGS.data_dir, FLAGS.testdata_name))

    dataset_train = tf.contrib.data.Dataset.from_tensor_slices((train_images, train_lables))
    dataset_test = tf.contrib.data.Dataset.from_tensor_slices((test_images, test_labels))

    dataset_train = dataset_train.batch(FLAGS.batch_size)
    dataset_train = dataset_train.shuffle(buffer_size=10000)
    # train_dataset = train_dataset.repeat(self.epochs)
    dataset_train = dataset_train.repeat()
    iterator = dataset_train.make_initializable_iterator()
    next_element_train = iterator.get_next()

    dataset_test = dataset_test.batch(1000)
    dataset_test = dataset_test.repeat()
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()

    # placeholder
    with tf.variable_scope('Inputs'):
        tf_images = tf.placeholder(tf.float32, shape=(None, FLAGS.height, FLAGS.width, FLAGS.channel),
                                   name='images_train') / 255
        tf_labels = tf.placeholder(tf.int32, shape=(None, 1), name='labels_train')

        tf_images_test = tf.placeholder(tf.float32, shape=(None, FLAGS.height, FLAGS.width, FLAGS.channel),
                                        name='images_test') / 255
        tf_labels_test = tf.placeholder(tf.int32, shape=(None, 1), name='labels_test')

    # build net
    print('=== build net ===')
    logits, endpoints = mobilenet_v1(tf_images,
                                     num_classes=FLAGS.n_labels,
                                     dropout_keep_prob=0.999,
                                     is_training=True,
                                     min_depth=8,
                                     depth_multiplier=1.0,
                                     conv_defs=None,
                                     prediction_fn=tf.contrib.layers.softmax,
                                     spatial_squeeze=True,
                                     reuse=None,
                                     scope='MobilenetV1'
                                     )

    cross_losses = tf.losses.sparse_softmax_cross_entropy(labels=tf_labels, logits=logits)
    losses = tf.reduce_mean(cross_losses)
    tf.summary.scalar('loss', losses)
    # TODO decay lr
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(losses)

    # accuracy
    logits_test, endpoints = mobilenet_v1(tf_images_test,
                                          num_classes=FLAGS.n_labels,
                                          dropout_keep_prob=1,
                                          is_training=False,
                                          min_depth=8,
                                          depth_multiplier=1.0,
                                          conv_defs=None,
                                          prediction_fn=tf.contrib.layers.softmax,
                                          spatial_squeeze=True,
                                          reuse=True,
                                          scope='MobilenetV1_prediction')
    accuracy, accuracy_op = tf.metrics.accuracy(labels=tf_labels_test,
                                                predictions=tf.argmax(logits_test, axis=1))
    tf.summary.scalar('accuracy', accuracy_op)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        #
        sess.run([iterator.initializer, iterator_test.initializer])

        writer = tf.summary.FileWriter('./log', sess.graph)  # write to file
        merged = tf.summary.merge_all()
        for i in range(FLAGS.maxstep):
            batch_images, batch_labels = sess.run(next_element_train)

            _, _loss = sess.run([train_op, losses],
                                feed_dict={tf_images: batch_images, tf_labels: batch_labels})

            if i % 50 == 0:
                batch_images_test, batch_labels_test = sess.run(next_element_test)

                _accuracy = sess.run(accuracy_op,
                                     feed_dict={tf_images_test: batch_images_test,
                                                tf_labels_test: batch_labels_test})
                # writer.add_summary(record, i)
                print('=== step:%s  accuracy: %s  loss: %s===' % (i, _accuracy, _loss))


if __name__ == '__main__':
    tf.app.run()
