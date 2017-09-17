# encoding: utf-8


'''reinforcement learning'''
from __future__ import absolute_import
import gym
import numpy as np
import matplotlib.pyplot as plt
from DQN_modified import DeepQNetwork

plot_data = []


def run():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]

    RL = DeepQNetwork(N_ACTIONS, N_STATES)

    step = 0
    for i in range(600):  # 玩300个回合
        # init env
        observation = env.reset()
        step_in = 0
        while True:
            # refresh env
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            # modify the reward
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            RL.store_transition(observation, action, r, observation_)

            if step > 200 and step % 5 == 0:
                RL.learn()

            if done:
                print('step_in:%s  reward:%s' % (step_in, reward))
                plot_data.append(step_in)
                break
            observation = observation_
            step += 1
            step_in += 1
    # end of game
    print('game over')
    # env.destroy()

    # plot_data = np.array(plot_data, dtype='float32')
    # plot_data = np.divide(plot_data, plot_data.max())
    print(plot_data)

def plot_step():
    # global plot_data
    # plot_data = np.array(plot_data, dtype='float32')
    # plot_data = np.divide(plot_data, plot_data.max())
    plot_data = np.random.random_sample((400,))
    plt.plot(np.arange(len(plot_data)), plot_data)
    plt.show()


if __name__ == '__main__':
    run()
