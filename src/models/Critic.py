import torch
import torch.nn as nn
from torch.nn import Linear


def train_critic(critic, data, optimizer):
    optimizer.zero_grad()
    observations = data["observations"]
    discounted_rewards = data["discounted_rewards"]

    values = critic(observations)
    loss = ((values - discounted_rewards) ** 2).mean()

    loss.backward()
    optimizer.step()
    return loss.item()


class Critic(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()
        self.layer1 = Linear(observation_dim, 32)
        self.layer2 = Linear(32, 32)
        self.layer3 = Linear(32, 32)
        self.layer4 = Linear(32, 1)

    def forward(self, observation):
        x = self.layer1(observation)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        value = self.layer4(x)
        return value
