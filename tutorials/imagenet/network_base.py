# encoding: utf-8

'''
base network
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple

Convd = namedtuple('Convd', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
# default convolution
_CONV_DEFS = [
    Convd(kernel=[3, 3], stride=2, depth=32),
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

DEFAULT_PADDING = 'SAME'
batchnorm_fused = True


# decorator
def layer(op):
    def wraper(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # print('layer_input', layer_input)
        output_layer = op(self, layer_input, *args, **kwargs)
        self.layers[name] = output_layer
        # 保存数据
        self.feed(output_layer)
        return self

    return wraper


class BaseNetwork(object):
    def __init__(self, inputs, trainable=True):
        self.trainable = trainable
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)

        self.setup()

    def setup(self):
        raise NotImplementedError('must implement setup function')

    def get_unique_name(self, prefix):
        '''
        获取唯一的name
        :param prefix: 
        :return: 
        '''
        return sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1

    def feed(self, *args):
        assert len(args) != 0
        # 清空
        self.terminals = []
        for fed_layer in args:
            try:
                is_str = isinstance(fed_layer, basestring)
            except:
                is_str = isinstance(fed_layer, str)

            if is_str:
                fed_layer = self.layers[fed_layer]

            self.terminals.append(fed_layer)
        return self

    @layer
    def separable_conv(self, inputs, k_h, k_w, c_o, stride, name, relu=True):
        with slim.arg_scope([slim.batch_norm], fused=batchnorm_fused):
            output = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  kernel_size=[k_h, k_w],
                                                  depth_multiplier=1,
                                                  stride=stride,
                                                  padding=DEFAULT_PADDING,
                                                  activation_fn=None,
                                                  # weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
                                                  biases_initializer=None,
                                                  trainable=self.trainable,
                                                  scope=name + '_depthwise',
                                                  )

            output = slim.convolution2d(output,
                                        num_outputs=c_o,
                                        kernel_size=[1, 1],
                                        stride=1,
                                        # padding=DEFAULT_PADDING,
                                        activation_fn=tf.nn.relu if relu else None,
                                        normalizer_fn=slim.batch_norm,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.004),
                                        biases_initializer=slim.init_ops.zeros_initializer(),
                                        trainable=self.trainable,
                                        scope=name + '_pointwise')

        return output

    @layer
    def convb(self, inputs, k_h, k_w, c_o, stride, name):
        with slim.arg_scope([slim.batch_norm], fused=batchnorm_fused):
            output = slim.convolution2d(inputs, c_o, kernel_size=[k_h, k_w],
                                        stride=stride,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.004),
                                        scope=name)
        return output

    @layer
    def max_pool(self, inputs, k_h, k_w, s_h, s_w, padding_name=DEFAULT_PADDING, name=None):
        return tf.nn.max_pool(inputs,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding_name,
                              name=name)

    @layer
    def concat(self, values, axis, name=None):
        return tf.concat(values=values, axis=axis, name=name)


def _reduced_kernel_size_for_small_input(inputs, kernel_size):
    '''
    如果输入的尺寸太小。kernel_size适应小尺寸输入
    :param inputs: 
    :param kernel_size: 
    :return: 
    '''
    shape = inputs.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def mobilenet_v1_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None
                      ):
    depth = lambda d: max(d * depth_multiplier, min_depth)
    end_points = {}
    if conv_defs is None:
        conv_defs = _CONV_DEFS

    net = inputs

    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            # TODO some op
            rate = 1
            current_stride = 1

            for i, _layer in enumerate(conv_defs):
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= _layer.stride
                else:
                    layer_stride = _layer.stride
                    layer_rate = 1
                    current_stride *= _layer.stride

                end_point_base = 'Conv2d_%s' % i

                if isinstance(_layer, Convd):
                    end_point = end_point_base
                    net = slim.conv2d(net,
                                      depth(_layer.depth),
                                      kernel_size=_layer.kernel,
                                      stride=_layer.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point
                                      )

                    end_points[end_point] = net
                    if final_endpoint == end_point:
                        return net, end_point

                elif isinstance(_layer, DepthSepConv):
                    end_point = end_point_base + '_depthwise'

                    net = slim.separable_conv2d(net,
                                                None,
                                                kernel_size=_layer.kernel,
                                                depth_multiplier=1.0,
                                                # rate=1,
                                                rate=layer_rate,
                                                # stride=_layer.stride,
                                                stride=layer_stride,
                                                normalizer_fn=slim.batch_norm,
                                                scope=end_point
                                                )

                    end_points[end_point] = net
                    if final_endpoint == end_point:
                        return net, end_point

                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net,
                                      depth(_layer.depth),
                                      kernel_size=[1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)

                    end_points[end_point] = net
                    if final_endpoint == end_point:
                        return net, end_points


def mobilenet_v1(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV1'):
    with tf.variable_scope(scope, 'Mobile_v1', [inputs, num_classes],
                           reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v1_base(inputs,
                                                scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier,
                                                conv_defs=conv_defs
                                                )
            with tf.variable_scope('Logits'):
                kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])

                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a')
                end_points['AvgPool_1a'] = net
                # 1 x 1 x 1024
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')

                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            end_points['Logits'] = logits

            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        return logits, end_points
