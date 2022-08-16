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
from torch.utils.tensorboard import SummaryWriter
import yaml

def run_agent(env, q_model, num_episodes):

    for episode in range(num_episodes):
        state_current = np.array([env.reset()])
        total_episode_rewards = 0
        frames = 0

        while True:
            env.render()
            frames += 1

            action = np.argmax(predict(q_model, state_current, 'cpu'))
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

    env_name = config['env']
    playback_buffer_size = config['playback_buffer_size']
    num_episodes = config['num_episodes']
    playback_sample_size = config['playback_sample_size']
    target_network_update_rate = config['target_network_update_rate']

    env = gym.make(config['env'])

    if config['data_dir'] != 'None':
        writer = SummaryWriter('./{}/dqn-playback_buff_sz-{}\
                -playback_sample_size-{}\
                -target_network_update-{}'.format(config['data_dir'], playback_buffer_size, playback_sample_size, target_network_update_rate),
                flush_secs = 1)
    else:
        writer = None

    agent = AgentDQN(
        env.action_space.n,
        env.observation_space.shape[0],
        playback_buffer_size,
        num_episodes,
        playback_sample_size,
        target_network_update_rate,
        writer,
        config['model_dir']
        )

    agent.learn(env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_search', default='None', type=str)
    parser.add_argument('--train', default='None', type=str)
    parser.add_argument('--run', default='None', type=str)
    args = parser.parse_args()

    if False:


        config = {
            "env": env,
            "playback_buffer_size": tune.choice([2048, 4096, 8192, 16384, 32768]),
            "num_episodes": 3000,
            "playback_sample_size": tune.choice([32, 64, 128, 256, 512]),
            "target_network_update_rate": tune.choice([1024, 2048, 4096, 8192, 16384])
        }

        reporter = CLIReporter( metric_columns=["reward", "training_iteration"])

        result = tune.run(
            train_dqn,
            name = 'dqn-tune',
            local_dir = 'runs',
            config=config,
            num_samples=16,
            stop={"training_iteration": 2000},
            progress_reporter=reporter)

    elif args.train != 'None':

        with open(args.train, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        train_dqn(config)

    elif args.run != 'None':

        env = gym.make('LunarLander-v2')
        q_model = FCNN(env.observation_space.shape[0], env.action_space.n)
        q_model.load_state_dict(torch.load(args.run))
        run_agent(env, q_model, 100)


if __name__ == "__main__":
    main()
