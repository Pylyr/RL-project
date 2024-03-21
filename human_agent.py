from agent import Agent


class HumanAgent(Agent):
    def __init__(self, env):
        self.env = env

    def choose_action(self):
        action = int(input("Enter your move: "))
        while action not in range(self.env.columns) or self.env.playable_rows[action] == -1:
            print("Invalid move. Try again.")
            action = int(input("Enter your move: "))
        return action
