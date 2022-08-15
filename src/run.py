from functools import partial
import argparse
import gym
import numpy as np
import torch
from dqn_agent import AgentDQN
from fcnn import FCNN
from nn_utils import predict
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.tensorboard import SummaryWriter

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


def train_dqn(config):

    env = config['env']
    playback_buffer_size = config['playback_buffer_size']
    num_episodes = config['num_episodes']
    playback_sample_size = config['playback_sample_size']
    target_network_update_rate = config['target_network_update_rate']
    writer = None

    #writer = SummaryWriter('runs/dqn-playback_buff_sz-{}\
            #-playback_sample_size-{}\
            #-target_network_update-{}'.format(playback_buffer_size, playback_sample_size, target_network_update_rate)
            #flush_secs = 1)

    agent = AgentDQN(
        env.action_space.n,
        env.observation_space.shape[0],
        playback_buffer_size,
        num_episodes,
        playback_sample_size,
        target_network_update_rate,
        writer)

    q_model = agent.learn(env)
    return q_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_search', default=True)
    parser.add_argument('--train', default=True)
    parser.add_argument('--num_episodes', default=1, type=int)
    parser.add_argument('--playback_buffer_size', default=1, type=int)
    parser.add_argument('--playback_sample_size', default=1, type=int)
    parser.add_argument('--target_network_update_rate', default=1, type=int)
    parser.add_argument('--file_path', default='None')

    args = parser.parse_args()

    if args.param_search:

        env = gym.make('LunarLander-v2')

        config = {
            "env": env,
            "playback_buffer_size": tune.choice([2048, 4096, 8192, 16384, 32768]),
            "num_episodes": 3000,
            "playback_sample_size": tune.choice([32, 64, 128, 256, 512]),
            "target_network_update_rate": tune.choice([1024, 2048, 4096, 8192, 16384])
        }

        scheduler = ASHAScheduler( metric="reward", mode="max", max_t=20)

        reporter = CLIReporter(
            metric_columns=["reward", "training_iteration"])

        result = tune.run(
            train_dqn,
            name = 'dqn-tune',
            local_dir = 'runs',
            config=config,
            num_samples=15,
            stop={"training_iteration": 2000},
            #scheduler=scheduler,
            progress_reporter=reporter)

    elif args.train:
        env = gym.make('LunarLander-v2')

        config = {
            "env": env,
            "playback_buffer_size": 8192,
            "num_episodes": 3000,
            "playback_sample_size": 64,
            "target_network_update_rate": 2048
        }

        q_model = train_dqn(config)
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
