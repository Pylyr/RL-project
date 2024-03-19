from typing import List, Tuple
import torch
import random
from collections import namedtuple


Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, state: torch.Tensor, action: int, reward, next_state: torch.Tensor, done):
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
