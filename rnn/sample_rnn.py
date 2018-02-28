#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 上午11:55
# @Author  : wudizhangzhi
# @File    : sample_rnn.py

import tensorflow as tf

flags = tf.flags

flags.DEFINE_string('filepath', 'pg.txt', '训练文件路径')

FLAGS = flags.FLAGS


print('== 开始 ==')
# 读取数据
with open(FLAGS.filepath, 'r') as f:
    data = f.read()

chars = list(set(data))
print('一共有 {} 个单词, 共 {} 个不重复单词'.format(len(data), len(chars)))


# loss function
def loss_func():
    pass


# sample


# train


