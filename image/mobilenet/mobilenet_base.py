#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 下午8:54
# @Author  : wudizhangzhi
import tensorflow as tf
from collections import namedtuple

slim = tf.contrib.slim

_Op = namedtuple('Op', ['op', 'params'])


def op(func, **params):
    return _Op(func, params=params)


def split_separable_conv2d(input_tensor,
                           num_outputs,
                           scope=None,
                           normalizer_fn=None,
                           stride=1,
                           rate=1,
                           endpoints=None,
                           use_explicit_padding=False
                           ):
    scope += '_'
    with tf.name_scope(scope):
        dw_scope = scope + 'depthwise'
        kernel_size = [3, 3]
        padding = 'SAME'
        net = slim.separable_conv2d(
            input_tensor,
            None,
            kernel_size=kernel_size,
            stride=stride,
            rate=rate,
            depth_multiplier=1,
            normalizer_fn=normalizer_fn,
            padding=padding,
            scope=dw_scope,
        )
        endpoints[dw_scope] = net

        pw_scope = scope + 'pointwise'
        net = slim.conv2d(
            net,
            num_outputs,
            [1, 1],
            stride=1,
            normalizer_fn=normalizer_fn,
            scope=pw_scope
        )
        endpoints[pw_scope] = net
    return net


if __name__ == '__main__':
    pass
