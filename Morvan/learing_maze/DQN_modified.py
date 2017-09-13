# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 mermory_size=500,
                 batch_size=32,
                 e_greedy_decay=None,
                 output_graph=False,
                 ):
        '''
        :param n_actions: 行动数量
        :param n_feature: 状态数量
        :param learning_rate: 
        :param reward_decay: 奖励衰减率
        :param e_greedy: 多大概率选择最优情况， 贪婪率？
        :param replace_target_iter: 替换target的参数的间隔
        :param mermory_size: 存储大小
        :param batch_size: 每次学习时候输入的样本数
        :param e_greedy_decay: e_greedy的增长率， 贪婪概率的缩小率
        :param output_graph: 是否输出图像
        '''
        self.n_actions = n_actions
        self.n_features = n_features
        self.LR = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = mermory_size
        self.batch_size = batch_size
        self.epsilon_increase = e_greedy_decay
        self.output_graph = output_graph

        self.epsilon = 0 if self.epsilon_increase else self.epsilon_max

        # learing counter
        self.learn_step_counter = 0

        # init memory
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))  # s, a, r, s_

        # consist of [target_net, evaluate_net]
        self._build_net()

        # 赋值operation
        e_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        t_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        self.target_param_replace_op = [tf.assign(t, e) for e, t in zip(e_param, t_param)]  # 将eval_net的参数赋值给 target_net

        self.sess = tf.Session()
        if self.output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)
            self.saver = tf.train.Saver()  # define a saver

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # placeholder
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')

        w_initilizer, b_initilizer = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
        # =======  eval net ========
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu,
                                 kernel_initializer=w_initilizer,
                                 bias_initializer=b_initilizer)
            self.q_eval = tf.layers.dense(e1, self.n_actions,
                                          kernel_initializer=w_initilizer,
                                          bias_initializer=b_initilizer)

        # =======  target net ========
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu,
                                 kernel_initializer=w_initilizer,
                                 bias_initializer=b_initilizer)
            self.q_target = tf.layers.dense(t1, self.n_actions,
                                            kernel_initializer=w_initilizer,
                                            bias_initializer=b_initilizer)

        # caculate cost
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_target, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            self.a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=self.a_indices)  # shape=(None, )
            # self.q_eval_wrt_a = tf.reduce_max(self.q_eval)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a), name='TD_error')

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.LR).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, training=True):
        observation = observation[np.newaxis, :]  # 为了feed_dict 因为 self.s.shape = [None, ]
        if np.random.uniform() < self.epsilon:
            if not training:
                self.play_with_experience()
            action_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace params
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_param_replace_op)
            print('\ntarget_params_replaced\n')

        # 从存储中取数据学习
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        feed_dict = {
            self.s: batch_memory[:, :self.n_features],
            self.a: batch_memory[:, self.n_features],
            self.r: batch_memory[:, self.n_features + 1],
            self.s_: batch_memory[:, -self.n_features:],
        }

        _, cost = self.sess.run([self.train_op, self.loss],
                                feed_dict=feed_dict)
        if self.output_graph:
            self.saver.save(self.sess, 'params', write_meta_graph=False)
        self.cost_his.append(cost)

        # increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increase if self.epsilon < self.epsilon_max \
            else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):

        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training step')
        plt.show()

    def play_with_experience(self):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, 'params')
