import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import input_data

flags = tf.app.flags
flags.DEFINE_integer('epoch_num', 2, '训练周期')
flags.DEFINE_integer('batch_size', 32, '训练样本大小')
flags.DEFINE_float('lr', 0.001, '学习率')
FLAGS = flags.FLAGS


def main(_):
    # load data
    dataset = input_data.read_data_sets('.', one_hot=True)
    dataset_train = dataset.train
    dataset_validation = dataset.validation
    dataset_test = dataset.test

    # build graph
    with tf.name_scope('Input'):
        tf_images = tf.placeholder(tf.float32, (None, 28 * 28))
        tf_labels = tf.placeholder(tf.float32, (None, 10))

    tf_images_reshaped = tf.reshape(tf_images, (-1, 28, 28, 1))

    # convd
    with tf.name_scope('Convolution_Layer_1'):
        conv1 = tf.layers.conv2d(
            tf_images_reshaped,
            filters=16,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # (-1, 28, 28, 16)

        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=2,
            strides=2,
        )  # (-1, 14, 14, 16)

    with tf.name_scope('Convolutional_Layer_2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # (-1, 14, 14, 32)

        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=2,
            strides=2,
        )  # (-1, 7, 7, 32)

    with tf.name_scope('Dense_Layer'):
        flat = tf.reshape(pool2, (-1, 7 * 7 * 32))
        output = tf.layers.dense(flat, 10)  # (-1, 10)

    with tf.name_scope('Loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_labels, logits=output)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('Accuracy'):
        _accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(tf_labels, axis=1),
                                                     predictions=tf.argmax(output, axis=1))
        tf.summary.scalar('accuracy', accuracy_op)

    with tf.name_scope('Train'):
        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    writer = tf.summary.FileWriter('./log')
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        step = 0
        sess.run(init_op)
        while dataset_train.epochs_completed < FLAGS.epoch_num:
            step += 1
            # fetch training data
            train_images, train_labels = dataset_train.next_batch(FLAGS.batch_size)

            _, train_loss, train_accuracy = sess.run([train_op, loss, accuracy_op],
                                                     feed_dict={
                                                         tf_images: train_images,
                                                         tf_labels: train_labels
                                                     })

            if step % 100 == 0:
                valid_accuracy, result = sess.run([accuracy_op, merged],
                                                  feed_dict={
                                                      tf_images: dataset_validation.images,
                                                      tf_labels: dataset_validation.labels,
                                                  })
                print("""
                =============================
                step: {}, ecpho:{} 
                train_loss: {}, train_accuracy: {}
                valid_accuracy: {}
                ============================
                """.format(
                    step,
                    dataset_train.epochs_completed,
                    train_loss,
                    train_accuracy,
                    valid_accuracy
                ))
                writer.add_summary(result, step)
                saver.save(sess, './save/', global_step=step, write_meta_graph=False)


if __name__ == '__main__':
    tf.app.run()
