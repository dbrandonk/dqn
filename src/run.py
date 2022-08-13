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
    parser.add_argument('--train', default=True)
    parser.add_argument('--num_episodes', default=1, type=int)
    parser.add_argument('--playback_buffer_size', default=1, type=int)
    parser.add_argument('--playback_sample_size', default=1, type=int)
    parser.add_argument('--target_network_update_rate', default=1, type=int)
    parser.add_argument('--file_path', default='None')

    args = parser.parse_args()

    if args.train:
        env = gym.make('LunarLander-v2')

        agent = AgentDQN(
            env.action_space.n,
            env.observation_space.shape[0],
            args.playback_buffer_size)

        q_model = agent.learn(
            env,
            args.num_episodes,
            args.playback_sample_size,
            args.target_network_update_rate)

        torch.save(q_model.state_dict(), '../checkpoint/q_model.pth')

    else:

        if args.file_path in ['None']:
            print('No file path specifed!')
        else:
            env = gym.make('LunarLander-v2')
            q_model = FCNN(env.observation_space.shape[0], env.action_space.n)
            q_model.load_state_dict(torch.load(args.file_path))
            run_agent(env, q_model, args.num_episodes)


if __name__ == "__main__":
    main()
