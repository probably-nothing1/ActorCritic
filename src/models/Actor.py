import gin
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from utils.utils import create_fully_connected_network


def train_actor(actor, data, optimizer):
    optimizer.zero_grad()
    observations = data["observations"]
    actions = data["actions"]
    discounted_rewards = data["discounted_rewards"]
    values = data["values"]

    _, policy = actor(observations)
    log_probs = policy.log_prob(actions)
    loss = -(log_probs * (discounted_rewards - values)).mean()

    entropy = policy.entropy().mean()

    loss.backward()
    optimizer.step()
    return loss.item(), entropy.item()


@gin.configurable
class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super().__init__()
        sizes = [observation_dim, *hidden_sizes, action_dim]
        self.fc_net = create_fully_connected_network(sizes)

    def forward(self, observation):
        logits = self.fc_net(observation)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return action, policy

    def get_best_action(self, observation):
        with torch.no_grad():
            logits = self.fc_net(observation)
            return torch.argmax(logits)
