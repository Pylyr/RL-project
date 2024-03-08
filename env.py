import random
import torch
import torch.optim as optim
import torch.nn as nn
from typing import List, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import math
import torch.nn.functional as F
from collections import namedtuple
import cProfile


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        self.rows = 6
        self.columns = 7
        self.board = [[0] * self.columns for _ in range(self.rows)]
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.rows, self.columns), dtype=int)
        self.playable_rows = [0] * self.columns

    def step(self, action):
        # Check if action is valid
        if self.playable_rows[action] == self.rows or self.done:
            return self.board, 0, True, {}

        # Find the next open row
        playable_row = self.playable_rows[action]
        self.board[playable_row][action] = self.current_player
        self.playable_rows[action] += 1

        reward, self.done = self.check_win(playable_row, action)
        self.current_player = 3 - self.current_player  # Switch player
        return self.board, reward, self.done, {}

    def reset(self):
        self.board = [[0] * self.columns for _ in range(self.rows)]
        self.playable_rows = [0] * self.columns
        self.current_player = 1
        self.done = False
        return self.board

    def render(self, mode='human'):
        fig, ax = plt.subplots()
        # Draw the board
        for x in range(self.columns):
            for y in range(self.rows):
                circle = patches.Circle(
                    (x, self.rows - y - 1), 0.45, fill=False, color='black', lw=2)
                if self.board[y][x] == 1:
                    circle = patches.Circle(
                        (x, self.rows - y - 1), 0.45, fill=True, color='red', lw=2)
                elif self.board[y][x] == 2:
                    circle = patches.Circle(
                        (x, self.rows - y - 1), 0.45, fill=True, color='yellow', lw=2)
                ax.add_patch(circle)
        plt.xlim(-0.5, self.columns-0.5)
        plt.ylim(-0.5, self.rows-0.5)
        ax.set_aspect('equal')
        plt.grid()
        plt.xticks(range(self.columns))
        plt.yticks(range(self.rows))
        plt.show()

    def check_win(self, row, col):
        # Check only from the last move's position
        # Horizontal, vertical, diagonal down, diagonal up
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1  # Count includes the last move
            # Check in the positive direction
            for i in range(1, 4):
                x, y = row + dx*i, col + dy*i
                if x < 0 or x >= self.rows or y < 0 or y >= self.columns or self.board[x][y] != self.current_player:
                    break
                count += 1
            # Check in the negative direction
            for i in range(1, 4):
                x, y = row - dx*i, col - dy*i
                if x < 0 or x >= self.rows or y < 0 or y >= self.columns or self.board[x][y] != self.current_player:
                    break
                count += 1
            if count >= 4:  # Found 4 in a row
                return 1, True

        # Check for draw
        if all([x == self.rows for x in self.playable_rows]):
            return 0.5, True
        return 0, False


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        return self.action_space.sample()


Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, state: torch.Tensor, action: torch.Tensor, reward, next_state: torch.Tensor, done):
        action_tensor = torch.tensor(
            [action], dtype=torch.long)  # Wrap action as a tensor
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(
            state, action_tensor, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # batched sampling
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, batch_size=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.batch_size = batch_size
        self.gamma = 0.999  # Discount factor

    def choose_action(self, state, explore=True) -> int:
        sample = random.random()
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > epsilon_threshold or not explore:
            with torch.no_grad():
                state = state.unsqueeze(0)
                decision = self.policy_net(state)
                return decision.max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.action_dim)

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Separate the components of each transition
        batch_states = torch.stack(batch.state).float()
        batch_actions = torch.stack(batch.action).view(-1, 1).long()
        batch_rewards = torch.tensor(batch.reward, dtype=torch.float)
        batch_next_states = torch.stack(batch.next_state).float()
        batch_dones = torch.tensor(batch.done, dtype=torch.float)

        # Calculate current Q-values from the policy_net
        current_q_values = self.policy_net(batch_states).gather(
            1, batch_actions).squeeze(1)

        # Calculate the maximum Q-value for the next states from the target_net
        next_state_values = self.target_net(
            batch_next_states).max(1)[0].detach()

        # Apply (1 - done) to zero out the values for terminal states
        next_state_values = next_state_values * (1 - batch_dones)

        # Compute the expected Q values for the current state-action pairs
        expected_q_values = (next_state_values * self.gamma) + batch_rewards

        # Compute loss
        loss = F.mse_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, env, n_games, print_interval):
        losses = []
        for episode in tqdm(range(n_games)):
            observation = env.reset()
            observation = torch.tensor(observation, dtype=torch.float)
            done = False
            while not done:
                action = self.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                next_observation = torch.tensor(
                    next_observation, dtype=torch.float)

                reward = torch.tensor([reward], dtype=torch.float)
                done_tensor = torch.tensor([done], dtype=torch.float)

                self.replay_buffer.push(
                    observation, action, reward, next_observation, done_tensor)
                observation = next_observation

                loss = self.optimize_model()
                if loss is not None:
                    losses.append(loss)

            if (episode + 1) % print_interval == 0 and len(losses) > 0:
                avg_loss = sum(losses[-print_interval:]) / \
                    len(losses[-print_interval:])
                print(f"Episode {episode + 1}: Average Loss = {avg_loss}")


# Initialize environment and agent
env = ConnectFourEnv()
replay_buffer = ReplayBuffer(10000)
agent = DQNAgent(env.observation_space.shape,
                 env.action_space.n, replay_buffer,
                 batch_size=1)

# Train the agent
agent.train(env, 1000, 50)
