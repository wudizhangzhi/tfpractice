# encoding: utf-8

import numpy as np
import pandas as pd


class RL:
    def __init__(self, actions, learing_rate=0.01, reward_deacy=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learing_rate
        self.gamma = reward_deacy
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exits(observation)
        if np.random.uniform() < self.epsilon:
            # action_name = self.q_table.iloc[str(observation), :].argmax()
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # 这样，数值相同时，就会随机采取行动
            action_name = state_action.argmax()

        else:
            action_name = np.random.choice(self.actions)
        return action_name

    def check_state_exits(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


# off-police
class QLearningTable(RL):
    def learn(self, s, a, r, s_):
        '''
        learing
        :param s: 当前状态
        :param a: 行动
        :param r: 反馈
        :param s_: 下一步的状态
        :return: 
        '''
        self.check_state_exits(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)


# on-plice
class Sarsa(RL):
    def learn(self, s, a, r, s_, a_):
        self.check_state_exits(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.ix[s_, a_]

        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)


class SarsaLambda(RL):
    def __init__(self, actions, learing_rate=0.01, reward_deacy=0.9, e_greedy=0.9, trace_deacy=0.9):
        super(SarsaLambda, self).__init__(actions, learing_rate=0.01, reward_deacy=0.9, e_greedy=0.9)
        # backward view, eligibility trace.
        self.lambda_ = trace_deacy  # 回顾奖励衰变率
        self.eligibility_trace = self.q_table.copy()

    def check_state_exits(self, state):
        if state not in self.q_table.index:
            to_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state

            )
            self.q_table = self.q_table.append(to_append)

            self.eligibility_trace = self.eligibility_trace.append(to_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exits(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.ix[s_, a_]

        error = q_target - q_predict

        # method 1
        # self.eligibility_trace.ix[s, :] += 1

        # method 2
        self.eligibility_trace.ix[s, :] = 0
        self.eligibility_trace.ix[s, a] = 1

        self.q_table += self.lr * error * self.eligibility_trace

        self.eligibility_trace *= self.lambda_ * self.gamma