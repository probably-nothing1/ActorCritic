import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn import Linear


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.layer1 = Linear(observation_dim, 32)
        self.layer2 = Linear(32, 32)
        self.layer3 = Linear(32, 32)
        self.layer4 = Linear(32, action_dim)

    def forward(self, observation):
        x = self.layer1(observation)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        logits = self.layer4(x)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return action
