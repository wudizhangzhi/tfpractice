#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/6 下午8:43
# @Author  : wudizhangzhi


"""
尝试将鹿鼎记内容转换
"""
import os
import random
import re
import collections

import redis
import tensorflow as tf
import numpy as np
import jieba
import jieba.posseg as pseg
import time
from zhon.hanzi import punctuation as zhon_punctuation
from string import punctuation as eng_punctunation

punctuation = zhon_punctuation + eng_punctunation

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
redis_client = redis.Redis(connection_pool=pool)

# FLAG
flags = tf.app.flags

flags.DEFINE_string('filepath', '/Users/zhangzhichao/Documents/鹿鼎记.txt', '文本文件路径')
flags.DEFINE_string('additional_dict', 'dict.txt', '附加的字典')
flags.DEFINE_string('redis_key_vocabulary', 'ludingji', 'redis存储字典的key的名称')
flags.DEFINE_string('redis_key_index', 'ludingji_index', 'redis存储全文索引列表的key的名称')
flags.DEFINE_string('encoding', 'gbk', '文件的编码格式')

flags.DEFINE_integer('batch_size', 32, '训练数据集大小')
flags.DEFINE_integer('window_width', 4, '选择的窗口宽度')
flags.DEFINE_integer('num_skips', 2, '每次 windows 中只随机取 num_skips 次 数值对')

FLAGS = flags.FLAGS

UNK_LIST = ['。', '.']


def pure_text(text):
    return re.sub(r"[^\u4e00-\u9fa5]+", '', text)


# 读取文件并处理为redis有序集合
def generate_vocabulary_redis_from_txt(filename, additional_dict, redis_key, encoding='gbk'):
    start = time.time()
    if os.path.exists(additional_dict):
        jieba.load_userdict(additional_dict)
    if not os.path.exists(filename):
        raise Exception('没有目标txt文件: {}'.format(filename))
    line_no = 0
    char_count = 0
    with open(filename, 'r', encoding=encoding) as f:
        for line in f.readlines():
            line_no += 1
            char_list = jieba.cut(pure_text(line), cut_all=False)
            for _char in char_list:
                frequent = redis_client.zincrby(redis_key, _char)
                print("词频 {}: {}".format(_char, int(frequent)))
                char_count += 1
    total = redis_client.zcard(redis_key)
    duration = time.time() - start
    print('用时: {}, 一共处理: {}行, 拆分为 {}个词, 共: {}个词'.format(duration, line_no, char_count, total))
    print('-' * 30)


def convert_txt_to_index_list(filename, additional_dict, redis_key_vocabulary, redis_key_index, encoding='gbk'):
    start = time.time()
    if os.path.exists(additional_dict):
        jieba.load_userdict(additional_dict)
    if not os.path.exists(filename):
        raise Exception('没有目标txt文件: {}'.format(filename))
    with open(filename, 'r', encoding=encoding) as f:
        for line in f.readlines():
            # TODO 是否考虑标点符号
            char_list = jieba.cut(pure_text(line), cut_all=False)
            for _char in char_list:
                index = redis_client.zrank(redis_key_vocabulary, _char)
                if index:
                    redis_client.rpush(redis_key_index, int(index))
                    print("添加 {} -> {}".format(_char, index))
    duration = time.time() - start
    length = redis_client.llen(redis_key_index)
    print('用时: {}, 列表长度: {}'.format(duration, length))
    print('-' * 30)


def generate_train_batch(target_inex, window_width, batch_size, num_skips, redis_key_index):
    """
    word2vec 中的做法是 设置一个数值 num_skips, 每次 windows 中只随机取 num_skips 次 数值对
    :param target_inex:
    :param window_width:
    :param batch_size:
    :param num_skips:
    :param redis_key_index:
    :return:
    """
    # e.g:[window, window, target, window, window, window]

    assert batch_size > 2 * window_width, Exception('训练数据大小必须大于window大小')
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    length = redis_client.llen(redis_key_index)

    count = 0  # 数据数量
    while count < batch_size:
        if target_inex + window_width + 1 > length or target_inex - window_width < 0:
            start = 0
        else:
            start = target_inex - window_width
        end = start + window_width + 1
        span = redis_client.lrange(redis_key_index, start, end)

        window_target = target_inex - start
        context_words = [w for w in range(len(span)) if w != window_target]
        words_to_use = random.sample(context_words, num_skips)
        for i, context_word in enumerate(words_to_use):
            batch[count] = span[context_word]
            labels[count, 0] = span[window_target]
            count += 1
            if count >= batch_size:
                break
        # 完成后向右移动一位
        target_inex += 1

    return batch, labels, target_inex


def debug_batch_labels(batch, labels):
    for i in range(FLAGS.batch_size):
        context = redis_client.zrange(FLAGS.redis_key_vocabulary, batch[i], batch[i])[0].decode('utf8')
        target = redis_client.zrange(FLAGS.redis_key_vocabulary, labels[i][0], labels[i][0])[0].decode('utf8')
        print('{} -> {}'.format(context, target))


def main(_):
    # 判断是否生成字典
    vocabulary_redis_exists = redis_client.exists(FLAGS.redis_key_vocabulary)
    if not vocabulary_redis_exists:
        generate_vocabulary_redis_from_txt(FLAGS.filepath, FLAGS.additional_dict, FLAGS.redis_key_vocabulary,
                                           FLAGS.encoding)
    else:
        print('redis中已经存在字典')

    # 处理全文章变为索引列表
    index_redis_exists = redis_client.exists(FLAGS.redis_key_index)
    if not index_redis_exists:
        convert_txt_to_index_list(FLAGS.filepath, FLAGS.additional_dict, FLAGS.redis_key_vocabulary,
                                  FLAGS.redis_key_index, FLAGS.encoding)
    else:
        print('redis中已经存在文章索引列表')

    # TODO 生成训练数据
    batch, labels, target_index = generate_train_batch(0, FLAGS.window_width, FLAGS.batch_size, FLAGS.num_skips,
                                                       FLAGS.redis_key_index)
    debug_batch_labels(batch, labels)

    # TODO 开始训练


if __name__ == '__main__':
    # read_text_from_file('/Users/zhangzhichao/Documents/鹿鼎记.txt', 'ludingji')
    tf.app.run()
