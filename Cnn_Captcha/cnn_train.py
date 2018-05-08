from generate_captcha import generate_gray_captcha, TOTAL_NUM, TEMPLATE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_integer('epoch_num', 2, '训练周期')
flags.DEFINE_integer('batch_size', 32, '训练样本大小')
flags.DEFINE_float('lr', 0.001, '学习率')
flags.DEFINE_float('lr_decay', 0.9, '学习率衰退率')
flags.DEFINE_integer('captcha_width', 160, '验证码宽')
flags.DEFINE_integer('captcha_height', 60, '验证码高')
flags.DEFINE_integer('char_num', 4, '验证码字符数')
FLAGS = flags.FLAGS


def text2vec(text):
    placeholder = np.zeros([FLAGS.char_num * TOTAL_NUM], dtype=np.int8)
    for i, c in enumerate(text):
        idx = i * TOTAL_NUM + TEMPLATE.index(c)
        placeholder[idx] = 1
    return placeholder


def vec2text(vec):
    pos_list = vec.nonzero()[0]
    return ''.join((TEMPLATE[pos % TOTAL_NUM] for pos in pos_list))


def next_batch(batch_size):
    batch_imgs = np.zeros([batch_size, FLAGS.captcha_width * FLAGS.captcha_height])
    batch_labels = np.zeros([batch_size, FLAGS.char_num * TOTAL_NUM], dtype=np.int8)
    i = 0
    while i < batch_size:
        img, text = generate_gray_captcha()
        if img.shape != (FLAGS.captcha_height, FLAGS.captcha_width):
            continue
        batch_imgs[i, :] = img.flatten()
        batch_labels[i, :] = text2vec(text)
        i += 1
    return batch_imgs, batch_labels

# inputdata
WIDTH, HEIGHT = FLAGS.captcha_width, FLAGS.captcha_height
with tf.name_scope('Input'):
    tf_images = tf.placeholder(dtype=tf.float32, shape=(None, WIDTH * HEIGHT))
    tf_labels = tf.placeholder(dtype=tf.int8, shape=(None, TOTAL_NUM * FLAGS.char_num))

    tf_images_reshaped = tf.reshape(tf_images, (-1, HEIGHT, WIDTH, 1))  # (-1, 60, 160, 1)


def build_graph():

    # convolution layer 1
    with tf.name_scope('Convolution Layer 1'):
        conv1 = tf.layers.conv2d(
            tf_images_reshaped,
            filters=16,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # (-1, 60, 160, 16)

        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        )  # (-1, 30, 80, 16)
    # convolution layer 2
    with tf.name_scope('Convolution Layer 2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # (-1, 30, 80, 32)

        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        )  # (-1, 15, 40, 32)
    # convolution layer 3
    with tf.name_scope('Convolution Layer 3'):
        conv3 = tf.layers.conv2d(
            pool2,
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # (-1, 15, 40, 64)

        pool3 = tf.layers.max_pooling2d(
            conv3,
            pool_size=(3, 2),
            strides=(3, 2),
            padding='same',
        )  # (-1, 5, 20, 64)

    # dense layers
    with tf.name_scope('Dense Layers'):
        reshaped = tf.reshape(pool3, (-1, np.prod(pool3.get_shape().as_list()[1:])))
        dense1 = tf.layers.dense(reshaped, 1024)

        output = tf.layers.dense(dense1, TOTAL_NUM * FLAGS.char_num)

    return output


def train():
    output = build_graph()
    with tf.name_scope('Loss'):
        loss = tf.losses.softmax_cross_entropy(logits=output, onehot_labels=tf_labels)

    with tf.name_scope('Accuracy'):
        pass


def main(_):
    images, labels = next_batch(1)
    print(images.shape)
    print(labels.shape)
    print(vec2text(labels[0]))


if __name__ == '__main__':
    tf.app.run()
