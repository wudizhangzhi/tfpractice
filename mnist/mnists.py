import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import input_data

flags = tf.app.flags
flags.DEFINE_integer('epoch_num', 2, '训练周期')
flags.DEFINE_integer('batch_size', 32, '训练样本大小')
flags.DEFINE_integer('predict_num', 10, '测试大小')
flags.DEFINE_string('file', '', '测试图片')
flags.DEFINE_float('lr', 0.001, '学习率')
flags.DEFINE_boolean('is_train', True, '是否训练')
FLAGS = flags.FLAGS

with tf.name_scope('Input'):
    tf_images = tf.placeholder(tf.float32, (None, 28 * 28))
    tf_labels = tf.placeholder(tf.float32, (None, 10))


def build_graph():
    # build graph
    tf_images_reshaped = tf.reshape(tf_images, (-1, 28, 28, 1))

    # convd
    with tf.name_scope('Convolution_Layer_1'):
        conv1 = tf.layers.conv2d(
            tf_images_reshaped,
            filters=16,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # (-1, 28, 28, 16)

        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=2,
            strides=2,
        )  # (-1, 14, 14, 16)

    with tf.name_scope('Convolutional_Layer_2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # (-1, 14, 14, 32)

        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=2,
            strides=2,
        )  # (-1, 7, 7, 32)

    with tf.name_scope('Dense_Layer'):
        flat = tf.reshape(pool2, (-1, 7 * 7 * 32))
        output = tf.layers.dense(flat, 10)  # (-1, 10)
    return output


def train():
    # load data
    dataset = input_data.read_data_sets('.', one_hot=True)
    dataset_train = dataset.train
    dataset_validation = dataset.validation
    dataset_test = dataset.test

    # build graph
    output = build_graph()

    # train
    with tf.name_scope('Loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_labels, logits=output)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('Accuracy'):
        _accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(tf_labels, axis=1),
                                                     predictions=tf.argmax(output, axis=1))
        tf.summary.scalar('accuracy', accuracy_op)

    with tf.name_scope('Train'):
        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    writer = tf.summary.FileWriter('./log')
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        step = 0
        sess.run(init_op)
        while dataset_train.epochs_completed < FLAGS.epoch_num:
            step += 1
            # fetch training data
            train_images, train_labels = dataset_train.next_batch(FLAGS.batch_size)

            _, train_loss, train_accuracy = sess.run([train_op, loss, accuracy_op],
                                                     feed_dict={
                                                         tf_images: train_images,
                                                         tf_labels: train_labels
                                                     })

            if step % 100 == 0:
                valid_accuracy, result = sess.run([accuracy_op, merged],
                                                  feed_dict={
                                                      tf_images: dataset_validation.images,
                                                      tf_labels: dataset_validation.labels,
                                                  })
                print("""
                =============================
                step: {}, ecpho:{} 
                train_loss: {}, train_accuracy: {}
                valid_accuracy: {}
                ============================
                """.format(
                    step,
                    dataset_train.epochs_completed,
                    train_loss,
                    train_accuracy,
                    valid_accuracy
                ))
                writer.add_summary(result, step)
                saver.save(sess, './save/', global_step=step, write_meta_graph=False)


def predict():
    # load data
    dataset = input_data.read_data_sets('.', one_hot=True)
    dataset_test = dataset.test
    test_images, test_labels = dataset_test.next_batch(batch_size=FLAGS.predict_num)

    # build graph
    output = build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./save/'))

        predicts = sess.run(output, feed_dict={
            tf_images: test_images
        })

    accuracy = np.sum(np.equal(np.argmax(predicts, axis=1),
                               np.argmax(test_labels, axis=1))) * 100 / FLAGS.predict_num
    print('正确率: {}'.format(accuracy))
    import matplotlib.pyplot as plt
    col_num = round(FLAGS.predict_num // 2)
    fig, ax = plt.subplots(2, col_num)
    index = 0
    for img, pred in zip(test_images, predicts):
        subfig = ax[index // col_num, index % col_num]
        subfig.imshow(img.reshape((28, 28)))
        subfig.set_title(np.argmax(pred))
        index += 1
    plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def inference(file):
    # trans file
    input = Image.open(file)
    input = trim(input)
    input.thumbnail((28, 28), Image.ANTIALIAS)
    print(input.size)
    # input = input.resize((28, 28))
    sample = np.zeros((28, 28))
    sample[:, :] = 255
    input = rgb2gray(np.array(input))
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            sample[i, j] = input[i, j]
    # plt.imshow(sample, cmap=plt.get_cmap('gray'))
    # plt.show()
    input = sample.reshape((1, 28 * 28))
    # predict
    # build graph
    output = build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./save/'))

        predicts = sess.run(output, feed_dict={
            tf_images: input
        })
    print('预测: {}'.format(np.argmax(predicts[0])))


def main(_):
    if FLAGS.is_train:
        train()
    else:
        if FLAGS.file:
            inference(FLAGS.file)
        else:
            predict()


if __name__ == '__main__':
    tf.app.run()
