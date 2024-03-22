from connect_4_env import ConnectFourEnv
import random


class Agent:
    def __init__(self, env: ConnectFourEnv, name: str = "Agent"):
        self.env = env
        self.name = name

    def choose_action(self):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env, name="RandomAgent")


    def choose_action(self):
        return random.choice(self.env.get_legal_actions())
