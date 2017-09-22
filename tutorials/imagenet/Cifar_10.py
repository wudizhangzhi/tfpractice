# encoding: utf-8

import sys

if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle
import tensorflow as tf
import numpy as np

import os
import re

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log',
                           """Directory where to save log. """)
tf.app.flags.DEFINE_string('data_dir', 'cifar-10-batches-py',
                           """Directory where to load data. """)
tf.app.flags.DEFINE_string('traindata_name', 'data_batch_*',
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
tf.app.flags.DEFINE_boolean('is_save', False,
                            """whether save params""")


class Cifar_10:
    def __init__(self, n_labels, batch_size=128, maxstep=8000, epochs=10, img_h=32, img_w=32, img_d=3, lr=0.001,
                 lr_decay_rate=0.9, decay_per_step=1000, data_dir='cifar-10-batches-py', traindata_name='data_batch_*',
                 testdata_name='test_batch',
                 namespace_file='batches.meta'):
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.IMG_D = img_d

        self.n_labels = n_labels
        self.LR = lr
        self.batch_size = batch_size
        self.epochs = epochs  # TODO delete
        self.maxstep = maxstep
        self.lr_decay_rate = lr_decay_rate
        self.decay_per_step = decay_per_step
        # data filename
        self.traindata_filelist = []
        assert os.path.exists(data_dir)
        filelist = os.listdir(data_dir)
        for filename in filelist:
            if re.match(traindata_name, filename):
                self.traindata_filelist.append(os.path.join(data_dir, filename))
        self.testdata_name = os.path.join(data_dir, testdata_name)
        self.namespace_file = os.path.join(data_dir, namespace_file)
        self._build_net()
        self.saver = tf.train.Saver()

    @classmethod
    def unpickle(cls, file):
        with open(file, 'rb') as fo:
            if sys.version_info.major == 3:
                d = pickle.load(fo, encoding='bytes')
            else:
                d = pickle.load(fo)
        return d

    def _build_net(self):
        print('=== start build graph ===')
        # placeholder
        with tf.variable_scope('Input'):
            self.tf_images = tf.placeholder(tf.float32, shape=[None, self.IMG_H, self.IMG_W, self.IMG_D],
                                            name='images') / 255
            self.tf_labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
            tf.summary.image('images', self.tf_images)
        # wieght, bias initilizer
        w_initilizer, b_initilizer = tf.truncated_normal_initializer(stddev=5e-2), tf.constant_initializer(0)

        # 第一层卷积层
        with tf.variable_scope('C_layer_1'):
            conv1 = tf.layers.conv2d(self.tf_images,
                                     filters=64,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     )  # shape [batch, height, width, 64]

            pool1 = tf.layers.max_pooling2d(conv1,
                                            pool_size=3,
                                            strides=2,
                                            padding='same')  # shape [batch, height/2, width/2, 64]
            # norm1
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
        # 第二层卷积层
        with tf.variable_scope('C_layer_2'):
            conv2 = tf.layers.conv2d(norm1,
                                     filters=64,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     )  # shape [batch, height/2, width/2, 128]
            # norm2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            pool2 = tf.layers.max_pooling2d(norm2,
                                            pool_size=3,
                                            strides=2,
                                            padding='same')  # shape [batch, height/4, width/4, 128]

        # reshape
        # h_w_d = tf.reduce_prod(shape[1:])
        h_w_d = self.IMG_H // 4 * self.IMG_W // 4 * 64
        pool2_reshape = tf.reshape(pool2, [-1, h_w_d])
        # 隐藏层1
        with tf.variable_scope('Hidden_layer_1'):
            hidden_1 = tf.layers.dense(pool2_reshape,
                                       384,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initilizer,
                                       bias_initializer=b_initilizer)

        # 隐藏层2
        with tf.variable_scope('Hidden_layer_2'):
            hidden_2 = tf.layers.dense(hidden_1,
                                       192,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initilizer,
                                       bias_initializer=b_initilizer)

        # 输出层
        with tf.variable_scope('Hidden_layer_3'):
            self.output = tf.layers.dense(hidden_2, self.n_labels,
                                          kernel_initializer=w_initilizer,
                                          bias_initializer=b_initilizer)

        # accuracy
        with tf.variable_scope('Accuracy'):
            accuracy, self.accuracy_op = tf.metrics.accuracy(labels=self.tf_labels,
                                                             predictions=tf.argmax(self.output, axis=1))
            tf.summary.scalar('accuracy', self.accuracy_op)  # add loss to scalar summary
        # loss
        with tf.variable_scope('Loss'):
            # self.losses = tf.losses.softmax_cross_entropy(logits=output, onehot_labels=self.tf_labels)
            cross_loss = tf.losses.sparse_softmax_cross_entropy(logits=self.output, labels=self.tf_labels)
            self.losses = tf.reduce_mean(cross_loss)
            tf.summary.scalar('loss', self.losses)  # add loss to scalar summary

        with tf.variable_scope('Train'):
            # global_step = tf.Variable(0, trainable=False)
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self.LR_decay_op = tf.train.exponential_decay(self.LR,
                                                          self.global_step,
                                                          self.decay_per_step,
                                                          self.lr_decay_rate,
                                                          staircase=True)
            self.train_op = tf.train.AdamOptimizer(self.LR_decay_op).minimize(self.losses, global_step=self.global_step)

        # init
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())  # the local var is for accuracy_op
        print('=== end build graph ===')

    @classmethod
    def _handler_images_data(cls, proto):
        # return np.vstack((proto[:, :1024], proto[:, 1024:1024*2], proto[:, 1024*2:1024*3])).T.reshape((-1, 32, 32, 3))
        return np.transpose(proto.reshape((-1, 3, 1024)), axes=[0, 2, 1]).reshape((-1, 32, 32, 3))

    def produce_test_data(self, testfilename):
        # test data
        d_data = self.unpickle(testfilename)
        batch_labels = d_data[b'labels']
        batch_images = d_data[b'data']

        assert len(batch_labels) == len(batch_images)
        batch_labels = np.array(batch_labels)[:, np.newaxis]
        batch_images = self._handler_images_data(batch_images)
        self.dataset_test = tf.contrib.data.Dataset.from_tensor_slices((batch_images, batch_labels))
        return self.dataset_test

    def produce_data(self, filelist, testfilename, label_name_file=None):
        assert isinstance(filelist, list)
        assert len(filelist) > 0
        labels_list = []
        images_list = []
        for filename in filelist:
            d_data = self.unpickle(filename)
            batch_labels = d_data[b'labels']
            batch_images = d_data[b'data']

            assert len(batch_labels) == len(batch_images)
            batch_labels = np.array(batch_labels)[:, np.newaxis]
            labels_list.append(batch_labels)
            batch_images = self._handler_images_data(batch_images)
            images_list.append(batch_images)

        images = np.vstack(images_list)
        labels = np.vstack(labels_list)
        # self.data_set = tf.contrib.data.Dataset.zip(dataset_list)
        self.data_set = tf.contrib.data.Dataset.from_tensor_slices((images, labels))

        # test data
        self.produce_test_data(testfilename)

        # 处理名称 label_names
        self.label_names = []
        if label_name_file:
            if not tf.gfile.Exists(label_name_file):
                print("can't find %s" % label_name_file)
            else:
                d_data = self.unpickle(label_name_file)
                self.label_names = d_data[b'label_names']

        return self.data_set, self.dataset_test, self.label_names

    def _inputs(self):
        print('=== start retrieve data ===')
        train_dataset, test_dataset, label_names = self.produce_data(self.traindata_filelist, self.testdata_name,
                                                                     self.namespace_file)

        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        # train_dataset = train_dataset.repeat(self.epochs)
        train_dataset = train_dataset.repeat()
        self.iterator = train_dataset.make_initializable_iterator()
        next_element = self.iterator.get_next()

        test_dataset = test_dataset.batch(1000)
        test_dataset = test_dataset.repeat()
        self.iterator_test = test_dataset.make_initializable_iterator()
        next_element_test = self.iterator_test.get_next()
        print('=== end retrieve data ===')
        return next_element, next_element_test

    def _train(self, is_save=False):
        next_element, next_element_test = self._inputs()

        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.log_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=FLAGS.maxstep),
                                                      tf.train.NanTensorHook(self.losses)]
                                               ) as self.sess:
            # init
            self.sess.run(self.init_op)
            # Initialize an iterator over a dataset with 10 elements.
            self.sess.run([self.iterator.initializer, self.iterator_test.initializer])
            writer = tf.summary.FileWriter('./log', self.sess.graph)  # write to file
            merged = tf.summary.merge_all()
            step = 0
            while not self.sess.should_stop():
                batch_images, batch_labels = self.sess.run(next_element)
                _, _loss = self.sess.run([self.train_op, self.losses], feed_dict={
                    self.tf_labels: batch_labels,
                    self.tf_images: batch_images,
                })
                if step % 50 == 0:
                    batch_image_test, batch_labels_test = self.sess.run(next_element_test)
                    _accuracy, record, _lr, _global_step = self.sess.run(
                        [self.accuracy_op, merged, self.LR_decay_op, self.global_step], feed_dict={
                            self.tf_labels: batch_labels_test,
                            self.tf_images: batch_image_test,
                        })
                    print('lr:%s step: %s, loss: %g , accuracy: %g' % (_lr, step, _loss, _accuracy))
                    writer.add_summary(record, _global_step)

                step += 1

        print('=== start training ===')

    def train(self, is_save=False):
        print('=== start retrieve data ===')
        # train_file_list = ['cifar-10-batches-py/data_batch_1', 'cifar-10-batches-py/data_batch_2']
        train_dataset, test_dataset, label_names = self.produce_data(self.traindata_filelist, self.testdata_name,
                                                                     self.namespace_file)

        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        # train_dataset = train_dataset.repeat(self.epochs)
        train_dataset = train_dataset.repeat()
        iterator = train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        test_dataset = test_dataset.batch(1000)
        test_dataset = test_dataset.repeat()
        iterator_test = test_dataset.make_initializable_iterator()
        next_element_test = iterator_test.get_next()
        print('=== end retrieve data ===')
        with tf.Session() as self.sess:
            # init
            self.sess.run(self.init_op)
            # Initialize an iterator over a dataset with 10 elements.
            self.sess.run([iterator.initializer, iterator_test.initializer])

            writer = tf.summary.FileWriter('./log', self.sess.graph)  # write to file
            merged = tf.summary.merge_all()

            print('=== start training ===')
            for step in range(self.maxstep):
                batch_images, batch_labels = self.sess.run(next_element)
                _, _loss = self.sess.run([self.train_op, self.losses], feed_dict={
                    self.tf_labels: batch_labels,
                    self.tf_images: batch_images,
                })
                if step % 50 == 0:
                    batch_image_test, batch_labels_test = self.sess.run(next_element_test)
                    _accuracy, record, _lr = self.sess.run([self.accuracy_op, merged, self.LR_decay_op], feed_dict={
                        self.tf_labels: batch_labels_test,
                        self.tf_images: batch_image_test,
                    })
                    print('lr:%s step: %s, loss: %g , accuracy: %g' % (_lr, step, _loss, _accuracy))
                    writer.add_summary(record, step)

            print('=== end training ===')
            print('=== save ===')
            if is_save:
                self.saver.save(self.sess, 'params', write_meta_graph=False)  # meta_graph not recommended

    def predict(self, images):
        self.saver.restore(self.sess, 'params')
        predicts = self.sess.run(self.output, feed_dict={self.tf_images: images})

        return [self.label_names[predict] for predict in predicts]

    @classmethod
    def plot_predicts(cls, images, predicts):
        import matplotlib.pyplot as plt
        num = len(images)
        assert len(predicts) == num
        img_per_row = num // 2
        fig, ax = plt.subplots(2, img_per_row)
        for i in range(img_per_row * 2):
            ax[i // img_per_row, i % img_per_row].imshow(images[i])
            ax[i // img_per_row, i % img_per_row].title(predicts[i])
        plt.show()

    def test_predict(self):
        self.produce_test_data('cifar-10-batches-py/test_batch')


def main(argv=None):
    c = Cifar_10(n_labels=FLAGS.n_labels, maxstep=FLAGS.maxstep, decay_per_step=FLAGS.decay_per_step, lr=FLAGS.lr)
    # c._train(is_save=FLAGS.is_save)
    c.train(is_save=FLAGS.is_save)


if __name__ == '__main__':
    tf.app.run()
