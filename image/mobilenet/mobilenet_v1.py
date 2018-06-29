#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/26 下午4:39
# @Author  : wudizhangzhi
import tensorflow as tf
from collections import namedtuple

# Conv defines 3x3 convolution layers
# DepthSepConv define 3x3 depthwise convolution followed by 1x1 pointwise convolution
# stride if the stride of the convolution
# depth is the number of channels or filters in a layer

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


def mobilenet_v1_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      use_explicit_padding=False,
                      scope=None):
    """
    Mobilenet v1
    
    Args:
        inputs: a tensor of shape [bathch_size, height, width, channel]
        final_endpoint: 
        min_depth: Minimum depth value(number of channels) for all convolution ops. actived when depth_multiplier < 1. 
                    and not actived when depth_multiplier >= 1.
        depth_multiplier: 
        conv_defs: a list of ConvDef namedtuple specifying the net architecture.
        output_stride:
        use_explicit_padding: padding use 'VALID' if true else 'SAME'
        scope: optional variable scope.
    Returns:
        pass
    Raise:
        
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier should be greater then zero')

    if not conv_defs:
        conv_defs = _CONV_DEFS

    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'

    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            # TODO
            current_stride = 1
            rate = 1

            net = inputs

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
                 scope='MobilenetV1',
                 global_pool=False):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, excepted 4, wat: %d' % len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v1_base(
                inputs,
                min_depth=min_depth,
                depth_multiplier=depth_multiplier,
                conv_defs=conv_defs,
                scope=scope,
            )

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


def mobilenet_v1_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False,
                           batch_norm_decay=0.9997,
                           batch_norm_epsilon=0.001):
    """defines the default MobilenetV1 arg scope.
    Args:
        is_training: Whether or not we're training the model. If this is set to
          None, the parameter is not added to the batch_norm arg_scope.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: The standard deviation of the trunctated normal weight initializer.
        regularize_depthwise: Whether or not apply regularization on depthwise.
        batch_norm_decay: Decay for batch norm moving average.
        batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    """
    # TODO
    batch_norm_params = {
        'center': True,
        'scale': True,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    if is_training:
        batch_norm_params['is_training'] = is_training

    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6,
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc
