from dqn_agent import AgentDQN
from fcnn import FCNN
from nn_utils import predict
import argparse
import gym
import numpy as np
import torch


def run_agent(env, q_model):

    NUM_EPISODES = 100

    for episode in range(NUM_EPISODES):
        state_current = np.array([env.reset()])
        total_episode_rewards = 0
        frames = 0

        while True:
            env.render()
            frames += 1

            action = np.argmax(predict(q_model, state_current))
            next_state, reward, done, info = env.step(action)
            total_episode_rewards = total_episode_rewards + reward
            state_current = np.array([next_state])

            if done:
                print(
                    f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} EPSILON: {0} FRAMES: {frames}')
                break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training")
    parser.add_argument("file_path")
    args = parser.parse_args()

    if args.training in ['None', 'True']:
        env = gym.make('LunarLander-v2')
        num_episodes = 3000

        agent = AgentDQN(env.action_space.n, env.observation_space.shape[0])
        q_model = agent.learn(env, num_episodes)
        torch.save(q_model.state_dict(), '../checkpoint/q_model.pth')

    elif args.training in ['False']:

        if args.file_path in ['None']:
            print('No file path specifed!')
            exit(0)
        else:
            env = gym.make('LunarLander-v2')
            q_model = FCNN(env.observation_space.shape[0], env.action_space.n)
            q_model.load_state_dict(torch.load(args.file_path))
            run_agent(env, q_model)
