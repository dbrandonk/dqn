from lunar_lander_net import LunarLanderNet
from dqn_run import dqn_runner
import gym

def main():
    env = gym.make('LunarLander-v2')
    dqn_runner(LunarLanderNet, env)

if __name__ == "__main__":
    main()

