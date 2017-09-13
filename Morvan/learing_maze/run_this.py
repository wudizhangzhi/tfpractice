# encoding: utf-8
import time
from maze_env import Maze
from RL_brain import *
from DQN_modified import DeepQNetwork


def update():
    for episode in range(100):
        # init obervation
        observation = env.reset()

        stepcounter = 0

        # start playing
        while True:
            # fresh env
            env.render()
            stepcounter += 1

            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                print('episode: %s, step: %s, reward: %s' % (episode + 1, stepcounter, reward))
                break

    print('game over')
    env.destroy()
    print(RL.q_table)


def update_sarsa():
    for episode in range(100):
        # init obervation
        observation = env.reset()

        stepcounter = 0

        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        # start playing
        while True:
            # fresh env
            env.render()
            stepcounter += 1

            # action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            action_ = RL.choose_action(str(observation_))

            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_

            if done:
                print('episode: %s, step: %s, reward: %s' % (episode + 1, stepcounter, reward))
                break

    print('game over')
    env.destroy()


def update_DQN():
    step = 0
    for episode in range(500):
        step_per = 0
        # initial observation
        observation = env.reset()
        while True:
            # refresh env
            env.render()

            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if step > 200 and step % 5 == 0:
                RL.learn()

            observation = observation_

            if done:
                print('total: %s, reward: %s' % (step_per, reward))
                break
            step += 1
            step_per += 1

    # end of game
    print('game over')
    env.destroy()


def play_once():
    step = 0
    # initial observation
    observation = env.reset()
    while True:
        # refresh env
        env.render()

        action = RL.choose_action(observation, training=False)
        # RL take action and get next observation and reward
        observation_, reward, done = env.step(action)

        RL.store_transition(observation, action, reward, observation_)

        observation = observation_

        if done:
            print('total: %s, reward: %s' % (step, reward))
            break
        step += 1
        time.sleep(1)
    # end of game
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    # RL = QLearningTable(actions=list(range(env.n_actions)))
    # RL = SarsaLambda(actions=list(range(env.n_actions)))
    RL = DeepQNetwork(n_actions=env.n_actions, n_features=env.n_features, output_graph=True)
    # 开始可视化环境 env
    # env.after(100, update)
    # env.after(100, update_sarsa)
    env.after(100, update_DQN)
    # env.after(100, play_once)
    env.mainloop()
    RL.plot_cost()