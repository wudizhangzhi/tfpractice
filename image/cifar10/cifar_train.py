import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import generate_data


flags = tf.app.flags
# string
flags.DEFINE_string('datapath', 'data', '数据路径')
# integer
flags.DEFINE_integer('epoch_num', 2, '训练周期')
flags.DEFINE_integer('train_step', 5000, '训练周期')
flags.DEFINE_integer('width', 32, '验证码宽')
flags.DEFINE_integer('height', 32, '验证码高')
flags.DEFINE_integer('classes', 10, '总类别')
flags.DEFINE_integer('batch_size', 64, '训练样本大小')
# float
flags.DEFINE_float('lr', 0.001, '学习率')
flags.DEFINE_float('lr_decay', 0.9, '学习率衰退率')
flags.DEFINE_float('keep_prob', 0.75, '保留率')
# boolean
flags.DEFINE_boolean('is_train', True, '是否是训练')
FLAGS = flags.FLAGS
