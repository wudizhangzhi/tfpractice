# encoding: utf-8

from six.moves import xrange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.data import Dataset

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001

# data
mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# placeholder
tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
images = tf.reshape(tf_x, [-1, 28, 28, 1])
tf_y = tf.placeholder(tf.int32, [None, 10])

# layers
conv1 = tf.layers.conv2d(  # shape (28, 28, 1)
    inputs=images,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu,
)  # shape (28, 28, 16)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=2,
    strides=2,
)  # shape (14, 14, 16)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=32,
    kernel_size=1,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)  # shape (14, 14, 32)

pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=2,
    strides=2,
)  # shape (7, 7, 32)

flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

output = tf.layers.dense(flat, 10)

loss = tf.losses.softmax_cross_entropy(logits=output, onehot_labels=tf_y)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph


from matplotlib import cm

try:
    from sklearn.manifold import TSNE

    HAS_SK = True
except:
    HAS_SK = False
    print('\nPlease install sklearn for layer visualization\n')


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


plt.ion()
for step in xrange(600):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:',  step, '| train loss: %0.4f' % loss_, '| test accuracy: %0.2f' % accuracy_)

        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
            labels = np.argmax(test_y, axis=1)[:plot_only]
            plot_with_labels(low_dim_embs, labels)

plt.ioff()

# print 10 predictions from test data
test_output = sess.run(output, feed_dict={tf_x: test_x[:10]})
pred_y = np.argmax(test_output, axis=1)
print('prediction numbers', pred_y)
print('real numbers', np.argmax(test_y[:10], axis=1))
