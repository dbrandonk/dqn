import copy
from collections import deque
import random
import numpy as np
import torch
from fcnn import FCNN
from nn_utils import train
from nn_utils import predict
from ray import tune

CURRENT_STATE_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
NEXT_STATE_INDEX = 3
DONE_INDEX = 4

class AgentDQN:
    def __init__(self, action_space, observation_space, playback_size, num_episodes, sample_batch_size, target_update_num_steps, writer):
        self.action_space = action_space
        self.observation_space = observation_space

        self.playback_buffer = deque(maxlen=playback_size)
        self.num_episodes = num_episodes
        self.sample_batch_size = sample_batch_size
        self.target_update_num_steps = target_update_num_steps
        self.writer = writer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q_model = FCNN(observation_space, action_space).to(self.device)
        self.target_q_model = FCNN(observation_space, action_space).to(self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.q_model.parameters(),
            lr=0.0001)

        self.epsilon = 1.0
        self.epsilon_reduction = 0.999
        self.gamma = 0.99

    def e_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(predict(self.q_model, state, self.device))
        return action

    def get_sample_batch(self):
        sample_batch = random.sample(self.playback_buffer, self.sample_batch_size)
        sample_batch = np.array(sample_batch, dtype=object)
        return sample_batch

    def update_network(self, sample_batch):

        current_states = np.vstack(sample_batch[:, CURRENT_STATE_INDEX])
        rewards = np.vstack(sample_batch[:, REWARD_INDEX])
        actions = np.vstack(sample_batch[:, ACTION_INDEX]).T[0]
        next_states = np.vstack(sample_batch[:, NEXT_STATE_INDEX])

        target_predictions = predict(self.target_q_model, next_states, self.device)
        max_actions = target_predictions.max(1)

        action_values = np.add(
            rewards.T, np.multiply(
                self.gamma, max_actions))[0]

        for sample_index in range(len(sample_batch)):
            if sample_batch[sample_index][DONE_INDEX]:
                action_values[sample_index] = sample_batch[sample_index][
                    REWARD_INDEX]

        model_predictons = predict(self.q_model, current_states, self.device)
        model_predictons[range(len(sample_batch)), actions] = action_values

        train(
            current_states,
            model_predictons,
            self.q_model,
            self.optimizer,
            self.criterion,
            self.device)

    def learn(self, env):

        steps = 0

        average_episode_reward = deque(maxlen=100)

        for episode in range(self.num_episodes):
            state_current = np.array([env.reset()])
            total_episode_rewards = 0.0
            frames = 0

            while True:
                steps += 1
                frames += 1

                action = self.e_greedy_action(state_current)

                next_state, reward, done, _ = env.step(action)
                next_state = np.array([next_state])
                total_episode_rewards += reward

                self.playback_buffer.append(
                    [state_current, action, reward, next_state, done])

                if len(self.playback_buffer) > self.sample_batch_size:
                    sample_batch = self.get_sample_batch()
                    self.update_network(sample_batch)

                state_current = next_state

                if (steps % self.target_update_num_steps) == 0:
                    self.target_q_model.load_state_dict(self.q_model.state_dict())

                if done:
                    average_episode_reward.append(total_episode_rewards)

                    if self.epsilon > 0.1:
                        self.epsilon = self.epsilon * self.epsilon_reduction

                    print(
                        f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} \
                                AVERAGE REWARD: {np.average(average_episode_reward)} \
                                EPSILON: {self.epsilon} FRAMES: {frames}')

                    #self.writer.add_scalar('avg_reward', np.average(average_episode_reward), episode)
                    tune.report(reward=np.average(average_episode_reward))

                    break

                if ((len(average_episode_reward) > 0) and (np.average(average_episode_reward) >= 200)):
                    return self.q_model

        return self.q_model
