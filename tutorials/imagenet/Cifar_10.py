# encoding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Cifar_10:
    def __init__(self, n_labels, batch_size=32, maxstep=8000, epochs=10, img_h=32, img_w=32, img_d=3, lr=0.9):
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.IMG_D = img_d

        self.n_labels = n_labels
        self.LR = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.maxstep = maxstep

        self.sess = tf.Session()
        self._build_net()

    def _build_net(self):
        print('=== start build graph ===')
        # placeholder
        self.tf_images = tf.placeholder(tf.float32, shape=[None, self.IMG_H * self.IMG_W * self.IMG_D])
        self.tf_labels = tf.placeholder(tf.int32, shape=[None, 1])

        # self.tf_labels = tf.reshape(self.tf_labels, [-1])
        self.tf_images_reshape = tf.reshape(self.tf_images, shape=[-1, self.IMG_H, self.IMG_W, self.IMG_D])
        # wieght, bias initilizer
        w_initilizer, b_initilizer = tf.random_normal_initializer(0.1, 0.3), tf.constant_initializer(0.1)

        # 第一层卷积层
        with tf.variable_scope('C_layer_1'):
            conv1 = tf.layers.conv2d(self.tf_images_reshape,
                                     filters=64,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     )  # shape [batch, height, width, 64]

            pool1 = tf.layers.max_pooling2d(conv1,
                                            pool_size=2,
                                            strides=2)  # shape [batch, height/2, width/2, 64]
            # norm1
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
        # 第二层卷积层
        with tf.variable_scope('C_layer_2'):
            conv2 = tf.layers.conv2d(norm1,
                                     filters=128,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     )  # shape [batch, height/2, width/2, 128]
            # norm2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            pool2 = tf.layers.max_pooling2d(norm2,
                                            pool_size=2,
                                            strides=2)  # shape [batch, height/4, width/4, 128]

        # reshape
        # h_w_d = tf.reduce_prod(shape[1:])
        h_w_d = self.IMG_H // 4 * self.IMG_W // 4 * 128
        pool2_reshape = tf.reshape(pool2, [-1, h_w_d])
        # 隐藏层1
        with tf.variable_scope('Hidden_layer_1'):
            hidden_1 = tf.layers.dense(pool2_reshape,
                                       128*2,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initilizer,
                                       bias_initializer=b_initilizer)

        # 输出层
        with tf.variable_scope('Hidden_layer_2'):
            output = tf.layers.dense(hidden_1, self.n_labels,
                                     kernel_initializer=w_initilizer,
                                     bias_initializer=b_initilizer)

        # accuracy
        with tf.variable_scope('Accuracy'):
            accuracy, self.accuracy_op = tf.metrics.accuracy(labels=self.tf_labels,
                                                             predictions=tf.argmax(output, axis=1))
            tf.summary.scalar('accuracy', self.accuracy_op)  # add loss to scalar summary
        # loss
        with tf.variable_scope('Loss'):
            # self.losses = tf.losses.softmax_cross_entropy(logits=output, onehot_labels=self.tf_labels)
            cross_loss = tf.losses.sparse_softmax_cross_entropy(logits=output, labels=self.tf_labels)
            self.losses = tf.reduce_mean(cross_loss)
            tf.summary.scalar('loss', self.losses)  # add loss to scalar summary

        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.losses)

        # init
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())  # the local var is for accuracy_op
        print('=== end build graph ===')

    def produce_data(self, filelist, testfilename, label_name_file=None):
        assert isinstance(filelist, list)
        assert len(filelist) > 0
        labels_list = []
        images_list = []
        for filename in filelist:
            with open(filename, 'rb') as f:
                d_data = pickle.load(f, encoding='bytes')
                batch_labels = d_data[b'labels']
                batch_images = d_data[b'data']

                assert len(batch_labels) == len(batch_images)
                batch_labels = np.array(batch_labels)[:, np.newaxis]
                labels_list.append(batch_labels)
                images_list.append(batch_images)
                # dataset = tf.contrib.data.Dataset.from_tensor_slices((batch_images, batch_labels))
                # dataset_list.append(dataset)

        images = np.vstack(images_list)
        labels = np.vstack(labels_list)
        # self.data_set = tf.contrib.data.Dataset.zip(dataset_list)
        self.data_set = tf.contrib.data.Dataset.from_tensor_slices((images, labels))

        # test data
        with open(testfilename, 'rb') as f:
            d_data = pickle.load(f, encoding='bytes')
            batch_labels = d_data[b'labels']
            batch_images = d_data[b'data']

            assert len(batch_labels) == len(batch_images)
            batch_labels = np.array(batch_labels)[:, np.newaxis]
            self.dataset_test = tf.contrib.data.Dataset.from_tensor_slices((batch_images, batch_labels))

        # 处理名称 label_names
        label_names = []
        if label_name_file:
            if not tf.gfile.Exists(label_name_file):
                print("can't find %s" % label_name_file)
            else:
                with open(label_name_file, 'rb') as f:
                    d_data = pickle.load(f, encoding='bytes')
                    label_names = d_data[b'label_names']

        return self.data_set, self.dataset_test, label_names

    def train(self, is_save=False):
        print('=== start retrieve data ===')
        train_file_list = ['cifar-10-batches-py/data_batch_1', 'cifar-10-batches-py/data_batch_2']
        train_dataset, test_dataset, label_names = self.produce_data(train_file_list, 'cifar-10-batches-py/test_batch',
                                                                     'cifar-10-batches-py/batches.meta')

        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.repeat(self.epochs)
        iterator = train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        test_dataset = test_dataset.batch(1000)
        test_dataset = test_dataset.repeat()
        iterator_test = test_dataset.make_initializable_iterator()
        next_element_test = iterator_test.get_next()
        print('=== end retrieve data ===')
        # init
        self.sess.run(self.init_op)
        # Initialize an iterator over a dataset with 10 elements.
        self.sess.run([iterator.initializer, iterator_test.initializer])

        writer = tf.summary.FileWriter('./log', self.sess.graph)  # write to file
        merged = tf.summary.merge_all()

        print('=== start training ===')
        for step in range(self.maxstep):
            batch_images, batch_labels = self.sess.run(next_element)
            self.sess.run(self.train_op, feed_dict={
                self.tf_labels: batch_labels,
                self.tf_images: batch_images,
            })
            if step % 50 == 0:
                batch_image_test, batch_labels_test = self.sess.run(next_element_test)
                _loss, _accuracy, record = self.sess.run([self.losses, self.accuracy_op, merged], feed_dict={
                    self.tf_labels: batch_labels_test,
                    self.tf_images: batch_image_test,
                })
                print('step: %s, loss: %g , accuracy: %g' % (step, _loss, _accuracy))
                writer.add_summary(record, step)

        print('=== end training ===')
        print('=== save ===')
        if is_save:
            saver = tf.train.Saver()  # define a saver
            saver.save(self.sess, 'params', write_meta_graph=False)  # meta_graph not recommended


if __name__ == '__main__':
    c = Cifar_10(n_labels=10)
    c.train()
