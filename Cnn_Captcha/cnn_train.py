from generate_captcha import generate_gray_captcha, TOTAL_NUM, TEMPLATE, rgb2gray
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

flags = tf.app.flags
flags.DEFINE_integer('epoch_num', 2, '训练周期')
flags.DEFINE_integer('train_step', 5000, '训练周期')
flags.DEFINE_integer('captcha_width', 160, '验证码宽')
flags.DEFINE_integer('captcha_height', 60, '验证码高')
flags.DEFINE_integer('char_num', 4, '验证码字符数')
flags.DEFINE_integer('batch_size', 64, '训练样本大小')
# float
flags.DEFINE_float('lr', 0.001, '学习率')
flags.DEFINE_float('lr_decay', 0.9, '学习率衰退率')
flags.DEFINE_float('keep_prob', 0.75, '保留率')
# boolean
flags.DEFINE_boolean('is_train', True, '是否是训练')
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
    tf_images = tf.placeholder(dtype=tf.float32, shape=(None, WIDTH * HEIGHT), name='input_images') / 255.
    tf_labels = tf.placeholder(dtype=tf.int8, shape=(None, TOTAL_NUM * FLAGS.char_num), name='input_labels')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

def build_graph():
    tf_images_reshaped = tf.reshape(tf_images, (-1, HEIGHT, WIDTH, 1))  # (-1, 60, 160, 1)
    # convolution layer 1
    with tf.name_scope('Convolution_Layer_1'):
        conv1 = tf.layers.conv2d(
            tf_images_reshaped,
            filters=16,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0.01),
            bias_initializer=tf.random_normal_initializer(mean=0.1)
        )  # (-1, 60, 160, 16)

        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        )  # (-1, 30, 80, 16)

        pool1 = tf.nn.dropout(pool1, keep_prob)
    # convolution layer 2
    with tf.name_scope('Convolution_Layer_2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0.01),
            bias_initializer=tf.random_normal_initializer(mean=0.1)
        )  # (-1, 30, 80, 32)

        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        )  # (-1, 15, 40, 32)
        pool2 = tf.nn.dropout(pool2, keep_prob)
    # convolution layer 3
    with tf.name_scope('Convolution_Layer_3'):
        conv3 = tf.layers.conv2d(
            pool2,
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0.01),
            bias_initializer=tf.random_normal_initializer(mean=0.1)
        )  # (-1, 15, 40, 64)

        pool3 = tf.layers.max_pooling2d(
            conv3,
            pool_size=(3, 2),
            strides=(3, 2),
            padding='same',
        )  # (-1, 5, 20, 64)
        pool3 = tf.nn.dropout(pool3, keep_prob)
    # dense layers
    with tf.name_scope('Dense_Layers'):
        reshaped = tf.reshape(pool3, (-1, np.prod(pool3.get_shape().as_list()[1:])))
        dense1 = tf.layers.dense(
            reshaped,
            1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0.01),
            bias_initializer=tf.random_normal_initializer(mean=0.1)
        )
        dense1 = tf.nn.dropout(dense1, keep_prob)

        output = tf.layers.dense(
            dense1,
            TOTAL_NUM * FLAGS.char_num,
            kernel_initializer=tf.random_normal_initializer(mean=0.01),
            bias_initializer=tf.random_normal_initializer(mean=0.1)
        )

    return output


def train():
    output = build_graph()
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=output, onehot_labels=tf_labels))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('Train_Op'):
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    with tf.name_scope('Accuracy'):
        tf_labels_reshaped = tf.reshape(tf_labels, (-1, TOTAL_NUM, FLAGS.char_num))
        output_reshaped = tf.reshape(output, (-1, TOTAL_NUM, FLAGS.char_num))
        _, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(tf_labels_reshaped, axis=2),
                                             predictions=tf.argmax(output_reshaped, axis=2))
        tf.summary.scalar('accuracy', accuracy_op)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./log')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(FLAGS.train_step):
            train_images, train_labels = next_batch(batch_size=FLAGS.batch_size)
            _, train_loss, train_accuracy, result = sess.run([train_op, loss, accuracy_op, merged],
                                                          feed_dict={
                                                            tf_images:train_images,
                                                            tf_labels:train_labels,
                                                            keep_prob: FLAGS.keep_prob
                                                          })
            if step % 100 == 0:
                validate_images, validate_lables = next_batch(100)
                validate_accuracy = sess.run(accuracy_op, feed_dict={
                    tf_images:validate_images,
                    tf_labels:validate_lables,
                    keep_prob:1,
                })

                print("""
                step: {}
                train_loss: {}, train_accuracy: {}
                validate_accuracy: {}
                ----------------------
                """.format(step, train_loss, train_accuracy, validate_accuracy))
                # save
                if validate_accuracy > 0.5:
                    saver.save(sess, './save/', global_step=step)
                # summary
                writer.add_summary(result, step)

def predict(image):
    output = build_graph()
    with tf.Session() as sess:
        saver.restore(sess, './save/')
        prediction = sess.run(ouput, feed_dict={
            tf_images:images,
            keep_prob:1
        })
    return prdiction

def predict_from_file(filepath):
    image = Image.open(filepath)
    img_array = np.array(img)
    img_array = rgb2gray(img_array) / 255.
    prediction = predict(img_array)

def main(_):
    if FLAGS.is_train:
        train()
    else:
        filepath = ''
        predict_from_file(filepath)


if __name__ == '__main__':
    tf.app.run()
