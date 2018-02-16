import gym
import argparse
from utils import to_discrete_value
from agent import Sarsa
from policy import Greedy, EpsGreedy


def main(args, agent):
    env = gym.make('CartPole-v0')

    for episode in range(args.episodes):
        observation = to_discrete_value(env.reset(), args.num_dizitized)
        done = False
        step = 0
        total_reward = 0
        while True:
            env.render()
            action = agent.get_action(observation)
            next_observation, reward, done, _ = env.step(action)
            next_observation = to_discrete_value(next_observation, args.num_dizitized)

            total_reward += reward
            print('episode:{}, step:{}, reward:{}, total reward:{}'.format(episode,
                                                                           step,
                                                                           reward,
                                                                           total_reward))

            update_info = {'state': observation,
                           'action': action,
                           'next_state': next_observation,
                           'reward': reward}
            agent.update(**update_info)

            observation = next_observation
            step += 1
            if done: break


    print('test start:', '-'*50)
    # episodeが終了したら5回テストしてみる
    # policyはgreedyにする
    agent.policy = Greedy()
    for test in range(5):
        observation = to_discrete_value(env.reset(), args.num_dizitized)
        done = False
        step = 0
        total_reward = 0
        while True:
            env.render()
            action = agent.get_action(observation)
            next_observation, reward, done, _ = env.step(action)
            next_observation = to_discrete_value(next_observation, args.num_dizitized)

            total_reward += reward
            print('episode:{}, step:{}, reward:{}, total reward:{}'.format(episode,
                                                                           step,
                                                                           reward,
                                                                           total_reward))
            observation = next_observation
            step += 1
            if done: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='コピペ')
    parser.add_argument('--episodes', type=int ,default=10)
    parser.add_argument('--num_dizitized', type=int, default=6)
    args = parser.parse_args()

    # agentを変更したい場合はここを編集
    policy = EpsGreedy(eps=0.5)
    agent = Sarsa(table_shape=(args.num_dizitized ** 4, 2), policy=policy, lr=0.01)
    main(args, agent)
