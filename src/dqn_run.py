import argparse
import gym
import numpy as np
from ray import tune
from ray.tune import CLIReporter
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
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

    playback_buffer_size = config['playback_buffer_size']
    num_episodes = config['num_episodes']
    playback_sample_size = config['playback_sample_size']
    target_network_update_rate = config['target_network_update_rate']

    env = config['env']
    model = config['model']

    try:
        writer = config['writer']
    except:
        writer = None

    agent = AgentDQN(
        env.action_space.n,
        env.observation_space.shape[0],
        playback_buffer_size,
        num_episodes,
        playback_sample_size,
        target_network_update_rate,
        writer,
        model
    )

    agent.learn(env)


def dqn_runner(model, env):

    parser = argparse.ArgumentParser()

    mx_group = parser.add_mutually_exclusive_group()
    mx_group.add_argument('--tune', default='None', type=str)
    mx_group.add_argument('--train', default='None', type=str)
    mx_group.add_argument('--run', default='None', type=str)

    parser.add_argument('--num_episodes', required=True, type=int)

    args = parser.parse_args()

    if args.tune != 'None':

        with open(args.tune, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        config['env'] = env
        config['model'] = model

        config['num_episodes'] = args.num_episodes
        config['playback_buffer_size'] = tune.choice(config['playback_buffer_size'])
        config['playback_sample_size'] = tune.choice(config['playback_sample_size'])
        config['target_network_update_rate'] = tune.choice(config['target_network_update_rate'])

        num_samples = config.pop('num_samples')

        reporter = CLIReporter(metric_columns=["reward", "training_iteration"])

        tune.run(
            train_dqn,
            name='dqn-tune',
            local_dir='data',
            config=config,
            num_samples=num_samples,
            stop={'training_iteration': args.num_episodes},
            progress_reporter=reporter)

    elif args.train != 'None':

        with open(args.train, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        writer = SummaryWriter(
                './{}/dqn-playback_buff_sz-{}-playback_sample_size-{}-target_network_update-{}'\
                .format('data', config['playback_buffer_size'], config['playback_sample_size'], config['target_network_update_rate']), flush_secs=1)

        config["writer"] = writer

        config['env'] = env
        config['model'] = model
        config["num_episodes"] = args.num_episodes
        train_dqn(config)

    elif args.run != 'None':

        q_model = model(env.observation_space.shape[0], env.action_space.n)
        q_model.load_state_dict(torch.load(args.run))
        run_agent(env, q_model, args.num_episodes)


if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    dqn_runner(FCNN, env)
