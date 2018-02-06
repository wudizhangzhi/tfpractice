# -*- coding:utf8 -*-
import numpy as np
import tensorflow as tf


# data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
baises = tf.Variable(tf.zeros([1]))

y = x_data * Weights + baises

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# end

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(baises))
        
sess.close()
