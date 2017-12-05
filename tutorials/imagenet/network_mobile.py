# encoding: utf-8
from network_base import BaseNetwork
import tensorflow as tf
import numpy as np


class MobileNet(BaseNetwork):
    def __init__(self, inputs, trainable=True):
        super(MobileNet, self).__init__(inputs, trainable)

    def setup(self):
        depth = lambda x: max(x, 8)
        with tf.variable_scope(None, 'MobilenetV1'):
            (self.feed('images')
             .convb(3, 3, 32, 2, name='Conv2d_0')
             .separable_conv(3, 3, depth(64), 1, name='Conv2d_1')
             .separable_conv(3, 3, depth(128), 2, name='Conv2d_2')
             .separable_conv(3, 3, depth(128), 1, name='Conv2d_3')
             .separable_conv(3, 3, depth(256), 2, name='Conv2d_4')
             .separable_conv(3, 3, depth(256), 1, name='Conv2d_5')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_6')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_7')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_8')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_9')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_10')
             .separable_conv(3, 3, depth(512), 1, name='Conv2d_11')
             )

            (self.feed('Conv2d_3').max_pool(2, 2, 2, 2, name='Conv2d_3_pool'))
            (self.feed('Conv2d_3_pool', 'Conv2d_7', 'Conv2d_11')
             .concat(3, name='feat_concat'))


def run_v1():
    from read_data import produce_test_data

    dataset_test = produce_test_data('cifar-10-batches-py/test_batch')
    # test_dataset = dataset_test.batch(10)
    # test_dataset = test_dataset.repeat()
    # iterator_test = test_dataset.make_initializable_iterator()
    # next_element_test = iterator_test.get_next()

    images = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
    # fix TypeError: Cannot create initializer for non-floating point type.
    # images = tf.to_float(images)
    # images feed next_element_test[0]

    # inputs = np.float32(next_element_test[0])
    with tf.Session() as sess:
        net = MobileNet({'images': images})
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        writer = tf.summary.FileWriter('./log', sess.graph)
        # merged = tf.summary.merge_all()

        re = sess.run(net.layers, feed_dict={images: dataset_test[0]})
        # sess.run(merged)


def run_v2():
    pass


if __name__ == '__main__':
    pass
