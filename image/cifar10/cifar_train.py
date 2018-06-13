import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import time
import datetime
from tensorflow.python.framework.errors_impl import OutOfRangeError

import generate_data

flags = tf.app.flags
# string
flags.DEFINE_string('datapath', 'data', '数据路径')
# integer
flags.DEFINE_integer('epoch_num', 10, '训练周期')
flags.DEFINE_integer('train_step', 5000, '训练周期')
flags.DEFINE_integer('width', 32, '宽')
flags.DEFINE_integer('height', 32, '高')
flags.DEFINE_integer('channel', 3, 'channels')
flags.DEFINE_integer('classes', 10, '总类别')
flags.DEFINE_integer('batch_size', 128, '训练样本大小')
flags.DEFINE_integer('predict_num', 10, '预测数量')
# float
flags.DEFINE_float('lr', 0.001, '学习率')
flags.DEFINE_float('lr_decay', 0.9, '学习率衰退率')
flags.DEFINE_float('lr_decay_step', 5000, '学习率衰退率')
flags.DEFINE_float('keep_prob', 0.75, '保留率')
# boolean
flags.DEFINE_boolean('is_train', True, '是否是训练')
flags.DEFINE_boolean('is_plt', False, '是否显示图表')
FLAGS = flags.FLAGS

# placeholder
tf_images = tf.placeholder(tf.float32, (None, FLAGS.width, FLAGS.height, FLAGS.channel))
tf_labels = tf.placeholder(tf.int32, (None,))
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')


def build_graph():
    w_alpha = 0.01
    b_alpha = 0.1
    # conv2d
    # w_c1 = tf.get_variable('w_c1', w_alpha * tf.random_normal([3, 3, 3, 32]))
    # b_c1 = tf.get_variable('b_c1', b_alpha * tf.random_normal([32]))
    w_c1 = tf.get_variable('w_c1', shape=[3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b_c1 = tf.get_variable('b_c1', shape=[32], initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(
        tf_images,
        filter=w_c1,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )  # (-1, 32 , 32 , 32)
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_c1))
    conv1 = tf.nn.max_pool(
        conv1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )  # (-1, 16, 16, 32)
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    # b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    w_c2 = tf.get_variable('w_c2', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b_c2 = tf.get_variable('b_c2', shape=[64], initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv2d(
        conv1,
        filter=w_c2,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )  # (-1, 16 ,16 ,64)
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_c2))
    conv2 = tf.nn.max_pool(
        conv2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )  # (-1, 8, 8, 64)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    # b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    w_c3 = tf.get_variable('w_c3', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b_c3 = tf.get_variable('b_c3', shape=[128], initializer=tf.constant_initializer(0.0))
    conv3 = tf.nn.conv2d(
        conv2,
        filter=w_c3,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )  # (-1, 8 ,8 ,128)
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b_c3))
    conv3 = tf.nn.max_pool(
        conv3,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )  # (-1, 4, 4, 128)
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # dense
    dim = np.prod(conv3.get_shape().as_list()[1:])
    conv3_flatten = tf.reshape(conv3, [-1, dim])
    # w_f1 = tf.Variable(w_alpha * tf.random_normal([dim, 1024]))
    # b_f1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    w_f1 = tf.get_variable('w_f1', shape=[dim, 1024], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b_f1 = tf.get_variable('b_f1', shape=[1024], initializer=tf.constant_initializer(0.0))
    dense = tf.nn.relu(tf.add(tf.matmul(conv3_flatten, w_f1), b_f1))
    dense = tf.nn.dropout(dense, keep_prob=keep_prob)

    # w_f2 = tf.Variable(w_alpha * tf.random_normal([1024, FLAGS.classes]))
    # b_f2 = tf.Variable(b_alpha * tf.random_normal([FLAGS.classes]))
    w_f2 = tf.get_variable('w_f2', shape=[1024, FLAGS.classes], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b_f2 = tf.get_variable('b_f2', shape=[FLAGS.classes], initializer=tf.constant_initializer(0.0))
    output = tf.add(tf.matmul(dense, w_f2), b_f2)
    return output


def train_data():
    filelist = os.listdir(FLAGS.datapath)
    train_filelist = [os.path.join(FLAGS.datapath, f) for f in filelist if f.startswith('data')]
    dataset_train, dataset_test, label_names = generate_data.produce_data(train_filelist,
                                                                          os.path.join(FLAGS.datapath, 'test_batch'),
                                                                          os.path.join(FLAGS.datapath,
                                                                                       'batches.meta'),
                                                                          )
    dataset_train = dataset_train.batch(FLAGS.batch_size)
    dataset_train = dataset_train.shuffle(buffer_size=10000)
    dataset_train = dataset_train.repeat(FLAGS.epoch_num)
    iterator = dataset_train.make_initializable_iterator()
    next_element = iterator.get_next()

    dataset_test = dataset_test.batch(1000)
    dataset_test = dataset_test.shuffle(buffer_size=1000)
    dataset_test = dataset_test.repeat()
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()
    print('导入数据完成')
    return next_element, iterator, next_element_test, iterator_test


def train():
    next_element, iterator, next_element_test, iterator_test = train_data()
    print('开始训练')
    output = build_graph()
    # TODO try one-hot
    # output_argmaxed = tf.argmax(output, axis=1)
    with tf.name_scope('Loss'):
        # loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=tf_labels))
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=output,
                                                                     labels=tf_labels))
        tf.summary.scalar('train_loss', loss)

    with tf.name_scope('Train'):
        global_step = tf.Variable(0, trainable=False)
        LR_decay_op = tf.train.exponential_decay(FLAGS.lr,
                                                 global_step,
                                                 FLAGS.lr_decay_step,
                                                 FLAGS.lr_decay,
                                                 staircase=True)

        train_op = tf.train.AdamOptimizer(LR_decay_op).minimize(loss, global_step=global_step)

    with tf.name_scope('Accuracy'):
        _, accuracy_op = tf.metrics.accuracy(labels=tf_labels,
                                             predictions=tf.argmax(output, axis=1))
        tf.summary.scalar('accuracy', accuracy_op)

    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log', sess.graph)
        saver = tf.train.Saver()
        start_time = datetime.datetime.now()
        sess.run(init_op)
        # Initialize an iterator over a dataset with 10 elements.
        sess.run([iterator.initializer, iterator_test.initializer])
        # while step < FLAGS.train_step:
        while True:
            try:
                # train batch data
                batch_image, batch_labels = sess.run(next_element)
            except OutOfRangeError:
                print('训练结束: 一共step：{}, 用时: {}'.format(_global_step,
                                                       datetime.datetime.now() - start_time))
                break

            _, _loss, _accuracy, _summary, _global_step = sess.run([train_op, loss,
                                                                    accuracy_op, merged,
                                                                    global_step],
                                                                   feed_dict={tf_images: batch_image,
                                                                              tf_labels: batch_labels,
                                                                              keep_prob: 0.9})
            if _global_step % 100 == 0:
                batch_image_test, batch_labels_test = sess.run(next_element_test)
                _accuracy_test, _lr = sess.run([accuracy_op, LR_decay_op],
                                               feed_dict={
                                                   tf_images: batch_image_test,
                                                   tf_labels: batch_labels_test,
                                                   keep_prob: 1
                                               })
                print('''
                --------------
                step: {}, lr: {}
                train_loss: {}, train_accuracy: {}
                test_accuracy: {}
                '''.format(_global_step, _lr, _loss, _accuracy, _accuracy_test))
                # summary
                writer.add_summary(_summary, _global_step)
                # save
                if _accuracy_test > 0.4:
                    saver.save(sess, './save/', global_step=global_step, write_meta_graph=False)
            # step += 1


def predict(images, test_labels=None, is_plt=False):
    print('开始预测')
    labels = generate_data.unpickle('data/batches.meta')[b'label_names']
    print(labels)
    import matplotlib.pyplot as plt
    output = build_graph()
    # 恢复session
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./save/'))
        predicts = sess.run(output, feed_dict={
            tf_images: images,
            keep_prob: 1
        })
        if is_plt:
            predict_map = zip(images, predicts)
            count = len(images)
            if count > 2:
                col_num = round(count // 2)
                fig, ax = plt.subplots(2, col_num)
                index = 0
                for img, pred in predict_map:
                    if labels:
                        label = labels[test_labels[index]]
                    else:
                        label = ''
                    fig_sub = ax[index // col_num, index % col_num]
                    fig_sub.imshow(img)
                    fig_sub.set_title('{}-{}'.format(labels[np.argmax(pred)], label))
                    index += 1
                plt.show()
            else:
                print(labels[np.argmax(predicts[0])])
        else:
            return predicts


def random_predict(num=6):
    test_images, test_lables = generate_data.produce_one_file('data/test_batch')
    total = len(test_images)
    indexes = np.random.choice(total, num, replace=False)
    choice_imgs = test_images[indexes]
    # choice_labels = test_lables[indexes]
    choice_labels = [test_lables[i] for i in indexes]

    results = predict(choice_imgs, choice_labels, is_plt=FLAGS.is_plt)
    accuracy = np.equal(np.argmax(results, axis=1), choice_labels)
    # print(accuracy)
    accuracy = np.sum(accuracy) * 100.0 / num
    print(accuracy)


def main(_):
    if FLAGS.is_train:
        train()
    else:
        random_predict(FLAGS.predict_num)


if __name__ == '__main__':
    tf.app.run()
