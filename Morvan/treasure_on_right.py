# encoding: utf-8

import numpy as np
import pandas as pd
import time

'''Q Learning example'''

np.random.seed(2)  # 伪随机


N_STATE = 6                     # 状态总数
ACTIONS = ['left', 'right']     # 动作表
EPSILON = 0.9                   # greedy plice 贪婪系数
REFRESH_TIME = 0.1                # 刷新环境时间
MAX_EPISODE = 13
GAMMA = 0.9                     # discount factor 类似于对未来的远见程度?
ALPHA = 0.1                     # learning rate


def build_q_table():
    table = pd.DataFrame(
        np.zeros((N_STATE, len(ACTIONS))),
        columns=ACTIONS,
    )
    return table


def choise_action(state, q_table):
    state_values = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_values.all() == 0):  # 如果随机数大于0.9即10%的概率随机选择行动 or 状态下所有值都是0
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_values.argmax()
    return action_name


def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATE - 2:  # 下一步就是终点
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = float(S/N_STATE)
    else:
        R = 0
        if S == 0:
            S_ = 0  # reach the wall
        else:
            S_ = S - 1

    return S_, R


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATE - 1) + ['T']  # --------T  environment
    # env_list = ['-'] * N_STATE + ['T']
    if S == 'terminal':
        interaction = '%s Episode: %s,  Total step: %s' % (' '*len(env_list), episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
    else:
        env_list[S] = 'O'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(REFRESH_TIME)


def rl():  # reinforcement learning
    q_table = build_q_table()
    for episode in range(MAX_EPISODE):
        # 初始状态
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choise_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ == 'terminal':
                is_terminated = True
                q_target = R
            else:
                q_target = R + GAMMA * q_table.iloc[S_, :].max()

            # 更新Q表
            q_table.ix[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            step_counter += 1
            update_env(S, episode, step_counter)
    return q_table



if __name__ == '__main__':
    table = rl()
    print()
    print(table)