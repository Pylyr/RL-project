import torch

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.counter = 0
        self.states = torch.empty((capacity, *state_shape), device=device)
        self.next_states = torch.empty((capacity, *state_shape), device=device)
        self.actions = torch.empty((capacity, 1), dtype=torch.long, device=device)
        self.rewards = torch.empty((capacity, 1), device=device)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=device)

    def push(self, state, action, reward, next_state, done):
        index = self.counter % self.capacity

        self.states[index] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[index] = torch.tensor([action], dtype=torch.long, device=self.device)
        self.rewards[index] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[index] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[index] = torch.tensor([done], dtype=torch.bool, device=self.device)

        self.counter += 1

    def sample(self, batch_size):
        max_index = min(self.counter, self.capacity)
        indices = torch.randint(0, max_index, (batch_size,), device=self.device)

        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices])

    def __len__(self):
        return min(self.counter, self.capacity)
