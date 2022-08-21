from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
from dqn_run import dqn_runner
from mario_net import MarioNet
import numpy as np

COMPRESSED_IMAGE_SZ = 84

class CompressedImageEnv():
    def __init__(self, env):

        self.STEP_AMOUNT = 4
        self.env = env
        self.action_space = env.action_space
        self.observation_space = [84,84]

        self.render = env.render
        self.close = env.close

    def reset(self):
        state = self.env.reset()

        #state_brg = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        state_gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state_gray_resize = cv2.resize(state_gray, (COMPRESSED_IMAGE_SZ, COMPRESSED_IMAGE_SZ))
        state_gray_resize_temp = np.expand_dims(state_gray_resize, axis=0)
        state_gray_resize = state_gray_resize_temp.copy()

        for step in range(self.STEP_AMOUNT - 1):
            state_gray_resize = np.vstack((state_gray_resize, state_gray_resize_temp))

        return state_gray_resize

    def step(self, action):

        state_gray_resize = None
        reward_total_step = 0

        for step in range(self.STEP_AMOUNT):
            state, reward, done, info = self.env.step(action)

            reward_total_step += reward

            state_gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state_gray_resize_temp = cv2.resize(state_gray, (COMPRESSED_IMAGE_SZ, COMPRESSED_IMAGE_SZ))
            state_gray_resize_temp = np.expand_dims(state_gray_resize_temp, axis=0)

            if not isinstance(state_gray_resize, np.ndarray):
                state_gray_resize = state_gray_resize_temp.copy()
            elif isinstance(state_gray_resize, np.ndarray):
                state_gray_resize = np.vstack((state_gray_resize, state_gray_resize_temp))

            if done:
                for step_done in range(step+1, self.STEP_AMOUNT):
                    state_gray_resize = np.vstack((state_gray_resize, state_gray_resize_temp))

                break



        return state_gray_resize, reward_total_step, done, info

def main():

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    #env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CompressedImageEnv(env)
    dqn_runner(MarioNet, env)

#    done = True
#    for step in range(5000):
#        if done:
#            state = env.reset()
#        state, reward, done, info = env.step(env.action_space.sample())

#        cv2.imshow('MARIO', state)
#        key = cv2.waitKey(500)

#    cv2.destroyAllWindows()
#    env.close()

if __name__ == "__main__":
    main()
