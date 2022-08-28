import copy
from collections import deque
import random
import numpy as np
import torch
from ray import tune
from nn_utils import train
from nn_utils import predict

CURRENT_STATE_INDEX = 0
ACTION_INDEX = 1
REWARD_INDEX = 2
NEXT_STATE_INDEX = 3
DONE_INDEX = 4


class AgentDQN:
    def __init__(
            self, action_space, observation_space, playback_size, num_episodes,
            sample_batch_size, target_update_num_steps, writer, model, dqn_train_rate, epsilon_reduction):

        self.action_space = action_space
        self.observation_space = observation_space

        self.playback_size = playback_size
        self.playback_buffer = None
        self.obs_collected = 0
        self.num_episodes = num_episodes
        self.sample_batch_size = sample_batch_size
        self.target_update_num_steps = target_update_num_steps
        self.writer = writer
        self.dqn_train_rate = dqn_train_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q_model = model(observation_space, action_space).to(self.device)
        self.target_q_model = model(
            observation_space,
            action_space).to(
            self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.q_model.parameters(),
            lr=0.0001)

        self.epsilon = 1.0
        self.epsilon_reduction = epsilon_reduction
        self.gamma = 0.99

    def _e_greedy_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(predict(self.q_model, state, self.device))
        return action

    def _get_sample_batch(self):
        sample_indexes = np.random.randint((self.playback_buffer.shape[0] - self.obs_collected), self.playback_buffer.shape[0], size = self.sample_batch_size)
        sample_batch = np.copy(self.playback_buffer[sample_indexes])
        return sample_batch

    def _update_network(self, sample_batch):

        current_states = np.stack(sample_batch[:, CURRENT_STATE_INDEX])
        rewards = np.vstack(sample_batch[:, REWARD_INDEX])
        actions = np.vstack(sample_batch[:, ACTION_INDEX]).T[0]
        next_states = np.stack(sample_batch[:, NEXT_STATE_INDEX])

        target_predictions = predict(
            self.target_q_model, next_states, self.device)
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
        top_avg_reward = 0

        for episode in range(self.num_episodes):

            state_current = env.reset()
            if not isinstance(state_current, np.ndarray):
                state_current = np.array([state_current])

            total_episode_rewards = 0.0
            frames = 0

            while True:
                steps += 1
                frames += 1

                action = self._e_greedy_action(state_current)

                next_state, reward, done, _ = env.step(action)
                if not isinstance(next_state, np.ndarray):
                    next_state = np.array([next_state])

                total_episode_rewards += reward

                self.obs_collected += 1
                if self.obs_collected > self.playback_size:
                    self.obs_collected = self.playback_size

                if (self.playback_buffer == None):
                    self.playback_buffer = np.array([state_current, action, reward, next_state, done], dtype=object)
                    self.playback_buffer = np.vstack([self.playback_buffer]*self.playback_size)
                else:
                    self.playback_buffer = np.vstack((self.playback_buffer, np.array([state_current, action, reward, next_state, done], dtype=object)))

                while(self.playback_buffer.shape[0] > self.playback_size):
                    self.playback_buffer = self.playback_buffer[1:]

                if ((self.obs_collected > self.sample_batch_size) and (steps % self.dqn_train_rate == 0)):
                    sample_batch = self._get_sample_batch()
                    self._update_network(sample_batch)


                state_current = next_state

                if (steps % self.target_update_num_steps) == 0:
                    self.target_q_model.load_state_dict(
                        self.q_model.state_dict())

                if done:
                    average_episode_reward.append(total_episode_rewards)

                    if self.epsilon > 0.1:
                        self.epsilon = self.epsilon * self.epsilon_reduction

                    avg_reward = np.average(average_episode_reward)

                    if ((len(average_episode_reward) == 1)
                            or (avg_reward > top_avg_reward)):
                        top_avg_reward = avg_reward
                        torch.save(
                            self.q_model.state_dict(),
                            './dqn-model-playback_buff_sz-{}-playback_sample_size-{}-target_network_update-{}.pth'
                            .format(self.playback_size, self.sample_batch_size, self.target_update_num_steps))

                    print (f'EPISODE: {episode} EPISODE AVG REWARD: {avg_reward} EPSILON: {self.epsilon} FRAMES: {frames}')

                    try:
                        self.writer.add_scalar(
                            'avg_reward', avg_reward, episode)
                    except BaseException:
                        pass

                    if tune.is_session_enabled():
                        tune.report(reward=avg_reward)

                    break

        env.close()
