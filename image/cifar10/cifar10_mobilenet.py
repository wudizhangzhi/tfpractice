from collections import namedtuple

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import time
import datetime
from tensorflow.python.framework.errors_impl import OutOfRangeError

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

# placeholder
tf_images = tf.placeholder(tf.float32, (None, FLAGS.width, FLAGS.height, FLAGS.channel))
tf_labels = tf.placeholder(tf.int32, (None,))
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

slim = tf.contrib.slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32, ),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),

    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),

    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024),
]

def _fixed_padding(inputs, kernel_size, rate=1):
    # TODO
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                    [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def _reduce_kernel_size_for_small_input(input_tensor, kernel_size):
    """
    如果input_tensor有dimension未知，则默认他足够大
    """
    tensor_shape = input_tensor.get_shape().as_list()
    if tensor_shape[1] is None or tensor_shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
            min(tensor_shape[1], kernel_size[0]),
            min(tensor_shape[2], kernel_size[1])
        ]
    return kernel_size_out


def mobile_net():
    final_endpoint = 'Conv2d_13_pointwise'
    min_depth = 8
    depth_multiplier = 1.0
    conv_defs = None
    output_stride = None
    use_explicit_padding = False
    scope = None
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier should be greater then zero')

    if not conv_defs:
        conv_defs = _CONV_DEFS

    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'

    with tf.variable_scope(scope, 'MobilenetV1', [tf_images]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            # TODO
            current_stride = 1
            rate = 1

            net = tf_images

            for i, conv_def in enumerate(conv_defs):
                print(i, conv_def)
                end_point_base = 'Conv2d_%d' % i

                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel)
                    print(net, depth(conv_def.depth), conv_def.kernel, conv_def.stride, end_point)
                    net = slim.conv2d(
                        net,
                        num_outputs=depth(conv_def.depth),
                        kernel_size=conv_def.kernel,
                        stride=conv_def.stride,
                        normalizer_fn=slim.batch_norm,
                        scope=end_point
                    )
                    print('===== conv2d =====')
                    end_points[end_point] = net

                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, DepthSepConv):  # depthwise + pointwise
                    end_point = end_point_base + '_depthwise'
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel, layer_rate)
                    # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/separable_conv2d
                    net = slim.separable_conv2d(
                        net,
                        num_outputs=None,  # If is None, then we skip the pointwise convolution stage.
                        kernel_size=conv_def.kernel,
                        depth_multiplier=1,
                        stride=layer_stride,
                        rate=layer_rate,
                        normalizer_fn=slim.batch_norm,
                        scope=end_point,
                    )
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(
                        net,
                        num_outputs=depth(conv_def.depth),
                        kernel_size=[1, 1],
                        normalizer_fn=slim.batch_norm,
                        scope=end_point,
                    )
                    end_points[end_point] = net

                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError('Unknow convolution type %s for layer %d' % (conv_def.ltype, i))

def build_graph():
    num_classes = 10
    dropout_keep_prob = 0.999
    is_training = True
    min_depth = 8
    depth_multiplier = 1.0
    conv_defs = None
    prediction_fn = tf.contrib.layers.softmax
    spatial_squeeze = True
    reuse = None
    scope = 'MobilenetV1'
    global_pool = False
    net, end_points = mobile_net()
    with tf.variable_scope('Logits'):
        #
        if global_pool:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            end_points['global_pool'] = net
        else:
            kernel_size = _reduce_kernel_size_for_small_input(net, [7, 7])
            net = slim.avg_pool2d(
                net,
                kernel_size=kernel_size,
                stride=1,
                padding='VALID',
                scope='AvgPool_1a'
            )
            end_points['AvgPool_1a'] = net

        if not num_classes:
            return net, end_points

        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                             scope='Conv2d_1c_1x1')
        print('net size: %s' % logits.get_shape())
        if spatial_squeeze:
            print('squeeze!!!!')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            print('net size: %s' % logits.get_shape())

    end_points['Logits'] = logits
    if prediction_fn:
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

def train_data():
    filelist = os.listdir(FLAGS.datapath)
    train_filelist = [os.path.join(FLAGS.datapath, f) for f in filelist if f.startswith('data')]
    dataset_train, dataset_test, label_names = generate_data.produce_data(train_filelist,
                                                                          os.path.join(FLAGS.datapath, 'test_batch'),
                                                                          os.path.join(FLAGS.datapath,
                                                                                       'batches.meta'),
                                                                          )
    dataset_train = dataset_train.batch(FLAGS.batch_size)
    dataset_train = dataset_train.shuffle(buffer_size=10000)
    dataset_train = dataset_train.repeat(FLAGS.epoch_num)
    iterator = dataset_train.make_initializable_iterator()
    next_element = iterator.get_next()

    dataset_test = dataset_test.batch(1000)
    dataset_test = dataset_test.shuffle(buffer_size=1000)
    dataset_test = dataset_test.repeat()
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()
    print('导入数据完成')
    return next_element, iterator, next_element_test, iterator_test


def train():
    next_element, iterator, next_element_test, iterator_test = train_data()
    print('开始训练')
    output, checkpoints = build_graph()
    # TODO try one-hot
    # output_argmaxed = tf.argmax(output, axis=1)
    with tf.name_scope('Loss'):
        one_hot_labels = tf.one_hot(tf.cast(tf_labels, dtype=tf.uint8), 10, dtype=tf.float32)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, output)
        total_loss = tf.losses.get_total_loss(name='total_loss')
        # loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=tf_labels))
        # loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=output,
        #                                                              labels=tf_labels))
        tf.summary.scalar('train_loss', total_loss)

    with tf.name_scope('Train'):
        global_step = tf.Variable(0, trainable=False)
        LR_decay_op = tf.train.exponential_decay(FLAGS.lr,
                                                 global_step,
                                                 FLAGS.lr_decay_step,
                                                 FLAGS.lr_decay,
                                                 staircase=True)

        train_op = tf.train.AdamOptimizer(LR_decay_op).minimize(total_loss, global_step=global_step)

    with tf.name_scope('Accuracy'):
        _, accuracy_op = tf.metrics.accuracy(labels=tf_labels,
                                             predictions=tf.argmax(output, axis=1))
        tf.summary.scalar('accuracy', accuracy_op)

    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log', sess.graph)
        saver = tf.train.Saver()
        start_time = datetime.datetime.now()
        sess.run(init_op)
        # Initialize an iterator over a dataset with 10 elements.
        sess.run([iterator.initializer, iterator_test.initializer])
        # while step < FLAGS.train_step:
        while True:
            try:
                # train batch data
                batch_image, batch_labels = sess.run(next_element)
            except OutOfRangeError:
                print('训练结束: 一共step：{}, 用时: {}'.format(_global_step,
                                                       datetime.datetime.now() - start_time))
                break

            _, _loss, _accuracy, _summary, _global_step = sess.run([train_op, loss,
                                                                    accuracy_op, merged,
                                                                    global_step],
                                                                   feed_dict={tf_images: batch_image,
                                                                              tf_labels: batch_labels,
                                                                              keep_prob: 0.9})
            if _global_step % 100 == 0:
                batch_image_test, batch_labels_test = sess.run(next_element_test)
                _accuracy_test, _lr = sess.run([accuracy_op, LR_decay_op],
                                               feed_dict={
                                                   tf_images: batch_image_test,
                                                   tf_labels: batch_labels_test,
                                                   keep_prob: 1
                                               })
                print('''
                --------------
                step: {}, lr: {}
                train_loss: {}, train_accuracy: {}
                test_accuracy: {}
                '''.format(_global_step, _lr, _loss, _accuracy, _accuracy_test))
                # summary
                writer.add_summary(_summary, _global_step)
                # save
                if _accuracy_test > 0.4:
                    saver.save(sess, './save/', global_step=global_step, write_meta_graph=False)
            # step += 1


def predict(images, test_labels=None, is_plt=False):
    print('开始预测')
    labels = generate_data.unpickle('data/batches.meta')[b'label_names']
    print(labels)
    import matplotlib.pyplot as plt
    output = build_graph()
    # 恢复session
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./save/'))
        predicts = sess.run(output, feed_dict={
            tf_images: images,
            keep_prob: 1
        })
        if is_plt:
            predict_map = zip(images, predicts)
            count = len(images)
            if count > 2:
                col_num = round(count // 2)
                fig, ax = plt.subplots(2, col_num)
                index = 0
                for img, pred in predict_map:
                    if labels:
                        label = labels[test_labels[index]]
                    else:
                        label = ''
                    fig_sub = ax[index // col_num, index % col_num]
                    fig_sub.imshow(img)
                    fig_sub.set_title('{}-{}'.format(labels[np.argmax(pred)], label))
                    index += 1
                plt.show()
            else:
                print(labels[np.argmax(predicts[0])])
        else:
            return predicts


def random_predict(num=6):
    test_images, test_lables = generate_data.produce_one_file('data/test_batch')
    total = len(test_images)
    indexes = np.random.choice(total, num, replace=False)
    choice_imgs = test_images[indexes]
    # choice_labels = test_lables[indexes]
    choice_labels = [test_lables[i] for i in indexes]

    results = predict(choice_imgs, choice_labels, is_plt=FLAGS.is_plt)
    accuracy = np.equal(np.argmax(results, axis=1), choice_labels)
    # print(accuracy)
    accuracy = np.sum(accuracy) * 100.0 / num
    print(accuracy)


def main(_):
    if FLAGS.is_train:
        train()
    else:
        random_predict(FLAGS.predict_num)


if __name__ == '__main__':
    tf.app.run()
