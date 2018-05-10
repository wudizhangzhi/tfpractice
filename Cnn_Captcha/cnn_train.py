from generate_captcha import generate_gray_captcha, TOTAL_NUM, TEMPLATE, rgb2gray
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

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
    # pos_list = vec.nonzero()[0]
    return ''.join((TEMPLATE[pos % TOTAL_NUM] for pos in vec))


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
    batch_size = FLAGS.batch_size
    w_alpha = 0.01
    b_alpha = 0.1
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.conv2d(
        tf_images_reshaped,
        filter=w_c1,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )  # (-1, 60 ,160 ,16)
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (-1, 30 , 80, 16)
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.conv2d(
        conv1,
        filter=w_c2,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )  # (-1, 30, 80, 32)
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (-1, 15, 40 ,32)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.conv2d(
        conv2,
        filter=w_c3,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )  # (-1, 15, 40, 64)
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (-1, 8, 20 ,64)
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # full connect
    dim = np.prod(conv3.get_shape().as_list()[1:])
    conv3_flatten = tf.reshape(conv3, (-1, dim))
    w_f1 = tf.Variable(w_alpha * tf.random_normal([dim, 1024]))
    b_f1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.nn.relu(tf.add(tf.matmul(conv3_flatten, w_f1), b_f1))
    dense = tf.nn.dropout(dense, keep_prob)

    w_f2 = tf.Variable(w_alpha * tf.random_normal([1024, FLAGS.char_num * TOTAL_NUM]))
    b_f2 = tf.Variable(b_alpha * tf.random_normal([FLAGS.char_num * TOTAL_NUM]))
    output = tf.add(tf.matmul(dense, w_f2), b_f2)

    ###############
    # with tf.name_scope('Convolution_Layer_1'):
    #     conv1 = tf.layers.conv2d(
    #         tf_images_reshaped,
    #         filters=16,
    #         kernel_size=3,
    #         strides=1,
    #         padding='same',
    #         activation=tf.nn.relu,
    #         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
    #         bias_initializer=tf.random_normal_initializer(stddev=0.1)
    #     )  # (-1, 60, 160, 16)
    #
    #     pool1 = tf.layers.max_pooling2d(
    #         conv1,
    #         pool_size=(2, 2),
    #         strides=(2, 2),
    #         padding='same',
    #     )  # (-1, 30, 80, 16)
    #
    #     pool1 = tf.nn.dropout(pool1, keep_prob)
    # # convolution layer 2
    # with tf.name_scope('Convolution_Layer_2'):
    #     conv2 = tf.layers.conv2d(
    #         pool1,
    #         filters=32,
    #         kernel_size=3,
    #         strides=1,
    #         padding='same',
    #         activation=tf.nn.relu,
    #         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
    #         bias_initializer=tf.random_normal_initializer(stddev=0.1)
    #     )  # (-1, 30, 80, 32)
    #
    #     pool2 = tf.layers.max_pooling2d(
    #         conv2,
    #         pool_size=(2, 2),
    #         strides=(2, 2),
    #         padding='same',
    #     )  # (-1, 15, 40, 32)
    #     pool2 = tf.nn.dropout(pool2, keep_prob)
    # # convolution layer 3
    # with tf.name_scope('Convolution_Layer_3'):
    #     conv3 = tf.layers.conv2d(
    #         pool2,
    #         filters=64,
    #         kernel_size=3,
    #         strides=1,
    #         padding='same',
    #         activation=tf.nn.relu,
    #         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
    #         bias_initializer=tf.random_normal_initializer(stddev=0.1)
    #     )  # (-1, 15, 40, 64)
    #
    #     pool3 = tf.layers.max_pooling2d(
    #         conv3,
    #         pool_size=(2, 2),
    #         strides=(2, 2),
    #         padding='same',
    #     )  # (-1, 8, 20, 64)
    #     pool3 = tf.nn.dropout(pool3, keep_prob)
    # # dense layers
    # with tf.name_scope('Dense_Layers'):
    #     reshaped = tf.reshape(pool3, (-1, np.prod(pool3.get_shape().as_list()[1:])))
    #     dense1 = tf.layers.dense(
    #         reshaped,
    #         1024,
    #         activation=tf.nn.relu,
    #         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
    #         bias_initializer=tf.random_normal_initializer(stddev=0.1)
    #     )
    #     dense1 = tf.nn.dropout(dense1, keep_prob)
    #
    #     output = tf.layers.dense(
    #         dense1,
    #         TOTAL_NUM * FLAGS.char_num,
    #         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
    #         bias_initializer=tf.random_normal_initializer(stddev=0.1),
    #     )
    ################
    # w_alpha = 0.01
    # b_alpha = 0.1
    # # 3 conv layer # 3 个 转换层
    # w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    # b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tf_images_reshaped, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    # conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv1 = tf.nn.dropout(conv1, keep_prob)
    #
    # w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    # b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    # conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv2 = tf.nn.dropout(conv2, keep_prob)
    #
    # w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    # b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    # conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    # conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3 = tf.nn.dropout(conv3, keep_prob)
    #
    # # Fully connected layer  # 最后连接层
    # w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    # b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    # dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    # dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    # dense = tf.nn.dropout(dense, keep_prob)
    #
    # # 输出层
    # w_out = tf.Variable(w_alpha * tf.random_normal([1024, TOTAL_NUM * FLAGS.char_num]))
    # b_out = tf.Variable(b_alpha * tf.random_normal([TOTAL_NUM * FLAGS.char_num]))
    # output = tf.add(tf.matmul(dense, w_out), b_out)

    return output


def train():
    output = build_graph()
    with tf.name_scope('Loss'):
        # loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=output, onehot_labels=tf_labels))
        loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=tf_labels))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('Train_Op'):
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    with tf.name_scope('Accuracy'):
        tf_labels_reshaped = tf.reshape(tf_labels, (-1, FLAGS.char_num, TOTAL_NUM))
        output_reshaped = tf.reshape(output, (-1, FLAGS.char_num, TOTAL_NUM))
        # _, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(tf_labels_reshaped, axis=2),
        #                                      predictions=tf.argmax(output_reshaped, axis=2))
        max_idx_labels = tf.argmax(tf_labels_reshaped, axis=2)
        max_idx_output = tf.argmax(output_reshaped, axis=2)
        correct_pred = tf.equal(max_idx_labels, max_idx_output)
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy_op)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./log')
    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(FLAGS.train_step):
            train_images, train_labels = next_batch(batch_size=FLAGS.batch_size)
            _, train_loss, train_accuracy, result = sess.run([train_op, loss, accuracy_op, merged],
                                                             feed_dict={
                                                                 tf_images: train_images,
                                                                 tf_labels: train_labels,
                                                                 keep_prob: FLAGS.keep_prob
                                                             })
            if step % 100 == 0:
                validate_images, validate_lables = next_batch(100)
                validate_accuracy = sess.run(accuracy_op, feed_dict={
                    tf_images: validate_images,
                    tf_labels: validate_lables,
                    keep_prob: 1,
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
    saver = tf.train.Saver()
    prediction = tf.reshape(output, (-1, FLAGS.char_num, TOTAL_NUM))
    prediction = tf.argmax(prediction, axis=2)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./save/'))
        prediction_results = sess.run(prediction, feed_dict={
            tf_images: image,
            keep_prob: 1
        })
        print('prediction_results: {}'.format(prediction_results))
        prediction_text_list = []
        for prediction_result in prediction_results:
            prediction_text_list.append(vec2text(prediction_result))
    return prediction_text_list


def predict_from_file(filepath):
    image = Image.open(filepath)
    img_array = np.array(image)
    img_array = rgb2gray(img_array) / 255.
    input = np.zeros([1, FLAGS.captcha_width * FLAGS.captcha_height])
    input[0, :] = img_array.flatten()
    prediction = predict(input)
    return prediction


def main(_):
    if FLAGS.is_train:
        train()
    else:
        print('开始预测')
        filepath = 'output.png'
        result = predict_from_file(filepath)
        print(result)


if __name__ == '__main__':
    tf.app.run()
