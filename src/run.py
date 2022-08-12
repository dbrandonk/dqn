import argparse
import gym
import numpy as np
import torch
from dqn_agent import AgentDQN
from fcnn import FCNN
from nn_utils import predict


def run_agent(env, q_model, num_episodes):

    for episode in range(num_episodes):
        state_current = np.array([env.reset()])
        total_episode_rewards = 0
        frames = 0

        while True:
            env.render()
            frames += 1

            action = np.argmax(predict(q_model, state_current))
            next_state, reward, done, _ = env.step(action)
            total_episode_rewards = total_episode_rewards + reward
            state_current = np.array([next_state])

            if done:
                print(
                    f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} \
                            EPSILON: {0} FRAMES: {frames}')
                break

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training")
    parser.add_argument("file_path")
    args = parser.parse_args()

    if args.training in ['None', 'True']:
        env = gym.make('LunarLander-v2')
        num_episodes = 3000

        playback_max_cap = 16384
        agent = AgentDQN(env.action_space.n, env.observation_space.shape[0], playback_max_cap)
        q_model = agent.learn(env, num_episodes)
        torch.save(q_model.state_dict(), '../checkpoint/q_model.pth')

    elif args.training in ['False']:

        if args.file_path in ['None']:
            print('No file path specifed!')
        else:
            env = gym.make('LunarLander-v2')
            q_model = FCNN(env.observation_space.shape[0], env.action_space.n)
            q_model.load_state_dict(torch.load(args.file_path))
            num_episodes = 100
            run_agent(env, q_model, num_episodes)


if __name__ == "__main__":
    main()
