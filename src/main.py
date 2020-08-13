# from models import create_actor, create_critic
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import Linear


class Actor(nn.Module):
    def __init__(self):
        self.layer1 = Linear(4, 32)
        self.layer2 = Linear(32, 32)
        self.layer3 = Linear(32, 32)
        self.layer4 = Linear(32, 2)

    def forward(self, observation):
        x = self.layer1(observation)
        x = F.tanh(x)
        x = self.layer2(observation)
        x = F.tanh(x)
        x = self.layer3(observation)
        x = F.tanh(x)
        logits = self.layer4(observation)
        policy = Categorical(logits)
        action = policy.sample()
        return logits, policy, action


class Critic(nn.Module):
    def __init__(self):
        self.layer1 = Linear(4, 32)
        self.layer2 = Linear(32, 32)
        self.layer3 = Linear(32, 32)
        self.layer4 = Linear(32, 2)

    def forward(self, observation):
        x = self.layer1(observation)
        x = F.tanh(x)
        x = self.layer2(observation)
        x = F.tanh(x)
        x = self.layer3(observation)
        value = F.tanh(x)
        return value


if __name__ == "__main__":
    # create wandb
    # create env
    actor = Actor()
    critic = Critic()
    # create exp buffer
    # create optimizers and train functions

    # train loop:
    #   1. simulate data
    #   2. train models
    #   3. log data
    pass
