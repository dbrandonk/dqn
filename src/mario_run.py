from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2

COMPRESSED_IMAGE_SZ = 84

class CompressedImageEnv():
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.reset = env.reset
        self.render = env.render
        self.close = env.close

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        #state_brg = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        state_gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state_gray_resize = cv2.resize(state_gray, (COMPRESSED_IMAGE_SZ, COMPRESSED_IMAGE_SZ))

        return state_gray_resize, reward, done, info

def main():

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CompressedImageEnv(env)

    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())

        cv2.imshow('MARIO', state)
        key = cv2.waitKey(500)

    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
