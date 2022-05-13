
def dqn_train_lunar_lander():

    lunar_lander = LunarLanderDQN()

    q_model = lunar_lander.dqn_learning()
    return q_model

def dqn_run_lunar_lander(q_model=None, weights_file=None):

    NUM_EPISODES = 100
    ACTION_SPACE = 4
    OBSERVATION_SPACE = 8

    if weights_file is not None:
        q_model = model_init(OBSERVATION_SPACE, ACTION_SPACE)
        q_model.load_weights(weights_file)

    env = gym.make('LunarLander-v2')

    for episode in range(NUM_EPISODES):
        state_current = np.array([env.reset()])
        total_episode_rewards = 0
        frames = 0



        while True:
            env.render()

            frames += 1

            action = np.argmax(q_model.predict(state_current))
            next_state, reward, done, info = env.step(action)
            total_episode_rewards = total_episode_rewards + reward
            next_state = np.array([next_state])
            state_current = next_state

            if done:
                print (f'EPISODE: {episode} EPISODE REWARD: {total_episode_rewards} EPSILON: {0} FRAMES: {frames}')
                break

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training")
    parser.add_argument("file_path")
    args = parser.parse_args()

    if args.training in ['None', 'True']:
        dqn_train_lunar_lander()
    elif args.training in ['False']:

        if args.file_path in ['None']:
            print('No file path specifed!')
            exit(0)

        dqn_run_lunar_lander(weights_file=args.file_path)
        exit(0)
    else:
        print('Unknown Error has occured. Ensure that arguments were entered correctly.')

env = gym.make('LunarLander-v2')
