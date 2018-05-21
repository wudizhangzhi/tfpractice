import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import generate_data

flags = tf.app.flags
# string
flags.DEFINE_string('datapath', 'data', '数据路径')
# integer
flags.DEFINE_integer('epoch_num', 10, '训练周期')
flags.DEFINE_integer('train_step', 5000, '训练周期')
flags.DEFINE_integer('width', 32, '验证码宽')
flags.DEFINE_integer('height', 32, '验证码高')
flags.DEFINE_integer('channel', 3, 'channels')
flags.DEFINE_integer('classes', 10, '总类别')
flags.DEFINE_integer('batch_size', 64, '训练样本大小')
# float
flags.DEFINE_float('lr', 0.001, '学习率')
flags.DEFINE_float('lr_decay', 0.9, '学习率衰退率')
flags.DEFINE_float('lr_decay_step', 5000, '学习率衰退率')
flags.DEFINE_float('keep_prob', 0.75, '保留率')
# boolean
flags.DEFINE_boolean('is_train', True, '是否是训练')
FLAGS = flags.FLAGS

# placeholder
tf_images = tf.placeholder(tf.float32, (None, FLAGS.width, FLAGS.height, FLAGS.channel))
tf_labels = tf.placeholder(tf.int8, (None, FLAGS.classes))
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')


def build_graph():
    w_alpha = 0.01
    b_alpha = 0.1
    # conv2d
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
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

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
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

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
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
    w_f1 = tf.Variable(w_alpha * tf.random_normal([dim, 1024]))
    b_f1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.nn.relu(tf.add(tf.matmul(conv3_flatten, w_f1), b_f1))
    dense = tf.nn.dropout(dense, keep_prob=keep_prob)

    w_f2 = tf.Variable(w_alpha * tf.random_normal([1024, FLAGS.classes]))
    b_f2 = tf.Variable(b_alpha * tf.random_normal([FLAGS.classes]))
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
    dataset_train = dataset_train.repeat()
    iterator = dataset_train.make_initializable_iterator()
    next_element = iterator.get_next()

    dataset_test = dataset_test.batch(1000)
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
        loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=tf_labels))
        tf.summary.scalar('train_loss', loss)

    with tf.name_scope('Train'):
        step = 0
        LR_decay_op = tf.train.exponential_decay(FLAGS.lr,
                                                 step,
                                                 FLAGS.lr_decay_step,
                                                 FLAGS.lr_decay,
                                                 staircase=True)

    train_op = tf.train.AdamOptimizer(LR_decay_op).minimize(loss)

    with tf.name_scope('Accuracy'):
        _, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(tf_labels, axis=1),
                                             predictions=tf.argmax(output, axis=1))
        tf.summary.scalar('accuracy', accuracy_op)

    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./log')
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session() as sess:
        sess.run(init_op)
        # Initialize an iterator over a dataset with 10 elements.
        sess.run([iterator.initializer, iterator_test.initializer])
        while step < FLAGS.train_step:
            # train batch data
            batch_image, batch_labels = sess.run(next_element)

            _, _loss, _accuracy, _summary = sess.run([train_op, loss, accuracy_op, merged],
                                                     feed_dict={tf_images: batch_image,
                                                                tf_labels: batch_labels,
                                                                keep_prob: 0.9})
            if step % 100 == 0:
                batch_image_test, batch_labels_test = sess.run(next_element_test)
                _accuracy_test = sess.run(accuracy_op, feed_dict={tf_images: batch_image_test,
                                                                  tf_labels: batch_labels_test,
                                                                  keep_prob: 1})
                print('''
                --------------
                step: {}
                train_loss: {}, train_accuracy: {}
                test_accuracy: {}
                '''.format(step, _loss, _accuracy, _accuracy_test))
                # summary
                writer.add_summary(_summary, step)
                # save
                if _accuracy_test > 0.5:
                    saver.save(sess, './save/', global_step=step)
            step += 1


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
