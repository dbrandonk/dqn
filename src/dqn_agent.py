from collections import deque
from fcnn import FCNN
from nn_utils import train
from nn_utils import predict
import copy
import gym
import numpy as np
import random
import torch

def e_greedy_action(q_model, action_space, state_current, epsilon):
    if np.random.random() < epsilon:
        action = np.random.randint(action_space)
    else:
        action = np.argmax(predict(q_model, state_current))
    return action

class AgentDQN:
    def __init__ (self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

        PLAYBACK_MAX_CAP = 4500
        self.playback_buffer = deque(maxlen=PLAYBACK_MAX_CAP)
        self.q_model = FCNN(observation_space, action_space)
        self.target_q_model = copy.deepcopy(self.q_model)
        self.criterion = torch.nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr = 0.001, weight_decay=0.001)
        self.optimizer = torch.optim.SGD(self.q_model.parameters(), lr=0.001, momentum=0.9)

        self.epsilon = 0.99
        self.epsilon_reduction = 0.999
        self.gamma = 0.99

    def get_sample_batch(self, sample_batch_size):
        sample_batch = random.sample(self.playback_buffer, sample_batch_size)
        sample_batch = np.array(sample_batch)
        return sample_batch

    def update_network(self, sample_batch):

        CURRENT_STATE_INDEX = 0
        ACTION_INDEX = 1
        REWARD_INDEX = 2
        NEXT_STATE_INDEX = 3

        current_states = np.vstack(sample_batch[:, CURRENT_STATE_INDEX])
        rewards = np.vstack(sample_batch[:, REWARD_INDEX])
        actions = np.vstack(sample_batch[:, ACTION_INDEX]).T[0]
        next_states = np.vstack(sample_batch[:, NEXT_STATE_INDEX])

        target_predictions = predict(self.target_q_model, next_states)
        max_actions = target_predictions.max(1)

        action_values = np.add(rewards.T, np.multiply(self.gamma, max_actions))[0]

        DONE_INDEX = 4
        for sample_index in range(len(sample_batch)):
            if sample_batch[sample_index][DONE_INDEX]:
                action_values[sample_index] = sample_batch[sample_index][REWARD_INDEX]

        model_predictons = predict(self.q_model, current_states)
        model_predictons[range(len(sample_batch)), actions] = action_values

        NUM_EPOCHS = 1
        train(NUM_EPOCHS, current_states, model_predictons, self.q_model, self.optimizer, self.criterion)

    def learn(self, env, num_episodes):

        SAMPLE_BATCH_SIZE = 32
        TARGET_UPDATE = 256
        steps = 0
        average_episode_reward = deque(maxlen=100)

        for episode in range(num_episodes):
            state_current = np.array([env.reset()])
            total_episode_rewards = 0
            frames = 0

            while True:
                steps += 1
                frames += 1

                action = e_greedy_action(self.q_model, self.action_space, state_current, self.epsilon)

                next_state, reward, done, info = env.step(action)
                next_state = np.array([next_state])
                total_episode_rewards = total_episode_rewards + reward

                self.playback_buffer.append([state_current, action, reward, next_state, done])

                if len(self.playback_buffer) > SAMPLE_BATCH_SIZE:
                    sample_batch = self.get_sample_batch(SAMPLE_BATCH_SIZE)
                    self.update_network(sample_batch)

                state_current = next_state

                if (steps % TARGET_UPDATE) == 0:
                    self.target_q_model = copy.deepcopy(self.q_model)

                if done:
                    average_episode_reward.append(total_episode_rewards)

                    if (self.epsilon > 0.1):
                        self.epsilon = self.epsilon*self.epsilon_reduction

                    print (f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} AVERAGE REWARD: {np.average(average_episode_reward)} EPSILON: {self.epsilon} FRAMES: {frames}')
                    break

        return self.q_model

