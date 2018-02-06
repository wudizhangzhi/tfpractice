#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/6 下午8:43
# @Author  : wudizhangzhi


"""
尝试将鹿鼎记内容转换
"""
import os
import re

import redis
import tensorflow as tf
import numpy as np
import jieba
import jieba.posseg as pseg
from zhon.hanzi import punctuation as zhon_punctuation
from string import punctuation as eng_punctunation

punctuation = zhon_punctuation + eng_punctunation

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
redis_client = redis.Redis(connection_pool=pool)


# FLAG



def pure_text(text):
    return re.sub(r"[^\u4e00-\u9fa5]+", '', text)


# 读取文件并处理为redis有序集合
def read_text_from_file(filename, redis_key, encoding='gbk'):
    if os.path.exists('dict.txt'):
        jieba.load_userdict('dict.txt')
    with open(filename, 'r', encoding=encoding) as f:
        for line in f.readlines():
            char_list = jieba.cut(pure_text(line), cut_all=False)
            for _char in char_list:
                frequent = redis_client.zincrby(redis_key, _char)
                print("{}: {}".format(_char, int(frequent)))


if __name__ == '__main__':
    read_text_from_file('/Users/zhangzhichao/Documents/鹿鼎记.txt', 'ludingji')
