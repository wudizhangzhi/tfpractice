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
import zipfile

import redis
import tensorflow as tf
import numpy as np
import jieba
import jieba.posseg as pseg
import time

from tensorflow.contrib.tensorboard.plugins import projector
from zhon.hanzi import punctuation as zhon_punctuation
from string import punctuation as eng_punctunation

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

punctuation = zhon_punctuation + eng_punctunation

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
redis_client = redis.Redis(connection_pool=pool)

# FLAG
flags = tf.app.flags
flags.DEFINE_string('mode', 'train', '模式')

flags.DEFINE_string('filepath', '/Users/zhangzhichao/Documents/鹿鼎记.txt', '文本文件路径')
flags.DEFINE_string('additional_dict', 'dict.txt', '附加的字典')
flags.DEFINE_string('redis_key_vocabulary', 'ludingji', 'redis存储字典的key的名称')
flags.DEFINE_string('redis_key_index', 'ludingji_index', 'redis存储全文索引列表的key的名称')
flags.DEFINE_string('encoding', 'gbk', '文件的编码格式')
flags.DEFINE_boolean('pure', True, '是否过滤')

flags.DEFINE_integer('batch_size', 32, '训练数据集大小')
flags.DEFINE_integer('window_width', 4, '选择的窗口宽度')
flags.DEFINE_integer('num_skips', 2, '每次 windows 中只随机取 num_skips 次 数值对')
flags.DEFINE_integer('valid_size', 16, '用于验证的数量')
flags.DEFINE_integer('valid_window', 1000, '用于验证的范围')
flags.DEFINE_integer('embedding_size', 128, 'Dimension, 每个单词的维度')
flags.DEFINE_integer('num_steps', 100000, '训练次数')
flags.DEFINE_string('font_path', '/Users/zhangzhichao/github/taidiiv2/taidii/public/font/simhei.ttf', '字体路径')
flags.DEFINE_integer('vocabulary_size', 50000, '字典大小。最常用的多少词')
flags.DEFINE_string('log_dir', 'log', '日志目录')

FLAGS = flags.FLAGS

UNK_LIST = ['。', '.']


def pure_text(text):
    return re.sub(r"[^\u4e00-\u9fa5]+", '', text)


# 读取文件并处理为redis有序集合
def generate_vocabulary_redis_from_txt(filename, additional_dict, redis_key, encoding='gbk', pure=True):
    start = time.time()
    if os.path.exists(additional_dict):
        jieba.load_userdict(additional_dict)
    if not os.path.exists(filename):
        raise Exception('没有目标文件: {}'.format(filename))
    line_no = 0
    char_count = 0
    with open(filename, 'r', encoding=encoding) as f:
        for line in f.readlines():
            line_no += 1
            if pure:
                line = pure_text(line)
                char_list = jieba.cut(line, cut_all=False)
            else:
                char_list = line.split()
            for _char in char_list:
                frequent = redis_client.zincrby(redis_key, _char)
                print("词频 {}: {}".format(_char, int(frequent)))
                char_count += 1
    total = redis_client.zcard(redis_key)
    duration = time.time() - start
    print('用时: {}, 一共处理: {}行, 拆分为 {}个词, 共: {}个词'.format(duration, line_no, char_count, total))
    print('-' * 30)


def convert_txt_to_index_list(filename, additional_dict, redis_key_vocabulary, redis_key_index, encoding='gbk',
                              pure=True):
    start = time.time()
    if os.path.exists(additional_dict):
        jieba.load_userdict(additional_dict)
    if not os.path.exists(filename):
        raise Exception('没有目标txt文件: {}'.format(filename))
    with open(filename, 'r', encoding=encoding) as f:
        for line in f.readlines():
            # TODO 是否考虑标点符号
            if pure:
                line = pure_text(line)
                char_list = jieba.cut(line, cut_all=False)
            else:
                char_list = line.split()
            for _char in char_list:
                index = int(redis_client.zrevrank(redis_key_vocabulary, _char))
                if index:
                    if index >= FLAGS.vocabulary_size:
                        continue
                    redis_client.rpush(redis_key_index, int(index))
                    print("添加 {} -> {}".format(_char, index))
    duration = time.time() - start
    length = redis_client.llen(redis_key_index)
    print('用时: {}, 列表长度: {}'.format(duration, length))
    print('-' * 30)


def generate_train_batch(target_inex, window_width, batch_size, num_skips, redis_key_index):
    """
    word2vec 中的做法是 设置一个数值 num_skips, 每次 windows 中只随机取 num_skips 次 数值对
            [context_word, target, context_word, context_word]
    :param target_inex:   目标的全文索引
    :param window_width:  窗口宽度
    :param batch_size:
    :param num_skips:  每次窗口随机取样的数量
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
        reach_end = target_inex + window_width + 1 > length
        if reach_end or target_inex - window_width < 0:
            start = 0
            if reach_end:
                target_inex = 0
        else:
            start = target_inex - window_width
        end = start + window_width + 1
        span = redis_client.lrange(redis_key_index, start, end)

        window_target = target_inex - start
        index_window_target_char = span[window_target]
        if int(index_window_target_char) > FLAGS.vocabulary_size:
            continue
        context_words = [w for w in range(len(span)) if w != window_target]
        words_to_use = random.sample(context_words, num_skips)
        for i, context_word in enumerate(words_to_use):
            context_word = span[context_word]
            if int(context_word) >= FLAGS.vocabulary_size:
                continue
            batch[count] = context_word
            labels[count, 0] = span[window_target]
            count += 1
            if count >= batch_size:
                break
        # 完成后向右移动一位
        target_inex += 1

    return batch, labels, target_inex


def debug_batch_labels(batch, labels):
    for i in range(FLAGS.batch_size):
        context = redis_client.zrevrange(FLAGS.redis_key_vocabulary, batch[i], batch[i])[0].decode('utf8')
        target = redis_client.zrevrange(FLAGS.redis_key_vocabulary, labels[i][0], labels[i][0])[0].decode('utf8')
        print('{} -> {}'.format(context, target))


def build_graph():
    vocabulary_size = FLAGS.vocabulary_size
    valid_examples = np.random.choice(100, FLAGS.valid_size, replace=False)

    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('Inputs'):
            # input data
            train_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])
            valid_dataset = tf.constant(valid_examples)

        with tf.device('/cpu:0'):
            with tf.name_scope('embedings'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform(
                        [vocabulary_size, FLAGS.embedding_size],
                        -1.0, 1.0)
                )

                embeded = tf.nn.embedding_lookup(embeddings, train_inputs)

            with tf.name_scope('weight'):
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [vocabulary_size, FLAGS.embedding_size],
                        stddev=1.0 / np.sqrt(FLAGS.embedding_size))
                )
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # TODO num_sampled?
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embeded,
                num_sampled=64,
                num_classes=vocabulary_size
            )
        )
        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        # 余弦相似性
        # (valid_size, embeddings_size) * (vocabulary_size, embeddings_size)^T
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

        init.run()
        print('Initialized')

        target_index = 0
        average_loss = 0
        # TODO
        sim_last = None
        embedings_last = None
        for step in range(FLAGS.num_steps):
            batch_train, labels_train, target_index = generate_train_batch(target_index, FLAGS.window_width,
                                                                           FLAGS.batch_size, FLAGS.num_skips,
                                                                           FLAGS.redis_key_index)
            feed_dict = {train_inputs: batch_train, train_labels: labels_train}
            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            _, loss_val, summary = session.run([optimizer, loss, merged], feed_dict=feed_dict)

            average_loss += loss_val
            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)

            # Add metadata to visualize the graph for the last run.
            if step == (FLAGS.num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # 查看相似的词组
            if step % 10000 == 0:
                sim = similarity.eval()
                # TODO
                if sim_last is not None:
                    print('是否一样: {}'.format(np.all(np.equal(sim, sim_last))))
                sim_last = sim
                if embedings_last is not None:
                    print('字典是否一样: {}'.format(session.run(tf.reduce_all(tf.equal(embeddings, embedings_last)))))
                embedings_last = embeddings

                for i in range(FLAGS.valid_size):
                    valid_word = redis_client.zrevrange(FLAGS.redis_key_vocabulary, valid_examples[i], valid_examples[i])[
                        0]
                    top_k = 8  # 最相似的8个

                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]

                    log_str = 'Nearest {}:'.format(valid_word.decode('utf8'))
                    for k in range(top_k):
                        close_word = redis_client.zrevrange(FLAGS.redis_key_vocabulary, nearest[k], nearest[k])[0]
                        log_str = '{} {},'.format(log_str, close_word.decode('utf8'))
                    print(log_str)


        final_embeddings = normalized_embeddings.eval()

        # Write corresponding labels for the embeddings.
        with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
            for i in range(vocabulary_size):
                f.write(redis_client.zrevrange(FLAGS.redis_key_vocabulary, i, i)[0].decode('utf8') + '\n')
        # Save the model for checkpoints.
        saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))
        # Create a configuration for visualizing embeddings with the labels in TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

        plot_samples(final_embeddings, vocabulary_size)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=FLAGS.font_path, size=14)
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=font)

    plt.savefig(filename)


def plot_samples(final_embeddings, vocabulary_size=5000):
    # simhei.ttf
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[np.random.randint(vocabulary_size, size=plot_only), :])
    labels = [redis_client.zrevrange(FLAGS.redis_key_vocabulary, i, i)[0].decode('utf8') for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)





def main(_):
    # 判断是否生成字典
    vocabulary_redis_exists = redis_client.exists(FLAGS.redis_key_vocabulary)
    if not vocabulary_redis_exists:
        generate_vocabulary_redis_from_txt(FLAGS.filepath, FLAGS.additional_dict, FLAGS.redis_key_vocabulary,
                                           FLAGS.encoding, FLAGS.pure)
    else:
        print('redis中已经存在字典')

    # 处理全文章变为索引列表
    index_redis_exists = redis_client.exists(FLAGS.redis_key_index)
    if not index_redis_exists:
        convert_txt_to_index_list(FLAGS.filepath, FLAGS.additional_dict, FLAGS.redis_key_vocabulary,
                                  FLAGS.redis_key_index, FLAGS.encoding, FLAGS.pure)
    else:
        print('redis中已经存在文章索引列表')

    # 生成训练数据
    # target_index = 0
    # batch, labels, target_index = generate_train_batch(target_index, FLAGS.window_width, FLAGS.batch_size, FLAGS.num_skips,
    #                                                    FLAGS.redis_key_index)
    # _batch = batch
    # debug_batch_labels(batch, labels)
    # batch, labels, target_index = generate_train_batch(target_index, FLAGS.window_width, FLAGS.batch_size,
    #                                                    FLAGS.num_skips,
    #                                                    FLAGS.redis_key_index)
    # debug_batch_labels(batch, labels)

    # TODO 开始训练

    # TODO 验证相近的词组
    build_graph()


if __name__ == '__main__':
    tf.app.run()
