import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class Actor(MLP):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__(state_dim, action_dim, hidden_dim)

class Critic(MLP):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__(input_dim, 1, hidden_dim)
