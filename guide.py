# encoding: utf-8

from six.moves import xrange

import tensorflow as tf
import numpy as np

'''https://www.tensorflow.org/programmers_guide/variables'''


def saving_variable():
    v1 = tf.Variable(tf.constant(12.0), dtype=tf.float32, name='v1')
    v2 = tf.Variable(tf.constant(9.0), dtype=tf.float32, name='v2')

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, 'logs/guide_model.ckpt')
        print("Model saved in file: %s" % save_path)


def restore_variable():
    v1 = tf.Variable(tf.constant(1.0), dtype=tf.float32, name='v1')
    v2 = tf.Variable(tf.constant(1.0), dtype=tf.float32, name='v2')

    saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # sess.run(init)
        # Restore variables from disk.
        saver.restore(sess, "logs/guide_model.ckpt")
        print("Model restored.")

        result = sess.run(tf.add(v1, v2))
        print('result : v1 + v2 = %s' % result)


def threading_and_queue():
    """
    tf.train.Coordinator: helps multiple threads stop together and report exceptions to a program that waits for them 
                            to stop
    tf.train.QueueRunner: create a number of threads cooperating to enqueue tensors in the same queue
    """
    # 1. define a queue and session
    queue = tf.FIFOQueue(10, dtypes=tf.float32)
    sess = tf.Session()

    # Main thread: create a coordinator.
    coord = tf.train.Coordinator()

    # 2. enqueue
    def enqueue_part():
        example = tf.random_uniform(shape=[4, 4])
        enqueue_op = queue.enqueue(example)
        qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        coord.join(enqueue_threads)

    # 3. dequeue
    def dequeue_part():
        def MyLoop(coord, queue):
            while not coord.should_stop():
                # if ...some condition...:
                val = queue.dequeue()
                if np.mean(val) > 0.6:
                    print('greater than 0.6')
                    coord.request_stop()

        # Create 5 threads that run 'MyLoop()'
        threads = [tf.threading.Thread(target=MyLoop, args=(coord, queue)) for i in xrange(5)]
        # Start the threads and wait for all of them to stop.
        for t in threads:
            t.start()
        coord.join(threads)

    enqueue_part()
    dequeue_part()

if __name__ == '__main__':
    # saving_variable()
    # restore_variable()
    threading_and_queue()