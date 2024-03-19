from env import ConnectFourEnv
import random


class Agent:
    def __init__(self, env: ConnectFourEnv):
        self.env = env

    def choose_action(self):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env
        self.name = "RandomAgent"

    def choose_action(self):
        return random.choice(self.env.get_legal_actions())
