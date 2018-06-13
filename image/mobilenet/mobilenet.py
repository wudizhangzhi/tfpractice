#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 下午8:54
# @Author  : wudizhangzhi
import tensorflow as tf
from collections import namedtuple

slim = tf.contrib.slim


def test():
    c1 = tf.constant(1.5)
    ret = tf.pow(tf.add(1.0, tf.exp(-1 * c1)), -1)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('log', sess.graph)
        result = sess.run(ret)
        print('result: %s' % result)
        writer.close()


if __name__ == '__main__':
    test()
