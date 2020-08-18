import gin
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.utils import create_fully_connected_network


def train_actor(actor, data, optimizer):
    optimizer.zero_grad()
    observations = data["observations"]
    actions = data["actions"]
    theta = data["theta"]

    _, policy = actor(observations)
    log_probs = policy.log_prob(actions)
    loss = -(theta * log_probs).mean()

    entropy = policy.entropy().mean()

    loss.backward()
    optimizer.step()
    return loss.item(), entropy.item()


@gin.configurable
def create_actor(observation_space, action_space, hidden_sizes):
    observation_dim = observation_space.shape[0]
    if isinstance(action_space, Discrete):
        action_dim = action_space.n
        return DiscreteActor(observation_dim, action_dim, hidden_sizes)
    elif isinstance(action_space, Box):
        action_dim = action_space.shape[0]
        return ContinuousActor(observation_dim, action_dim, hidden_sizes)
    else:
        raise ValueError(f"Unrecognized Action Space type: {type(action_space)}")


class DiscreteActor(nn.Module):
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


class ContinuousActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super().__init__()
        backbone_sizes = [observation_dim, *hidden_sizes[:-1]]
        self.backbone = create_fully_connected_network(backbone_sizes, output_activation_fn=nn.Tanh)

        head_sizes = [hidden_sizes[-1], action_dim]
        self.means_head = create_fully_connected_network(head_sizes)
        self.log_stds_head = create_fully_connected_network(head_sizes)

    def forward(self, observation):
        backbone_output = self.backbone(observation)
        means = self.means_head(backbone_output)
        log_stds = self.log_stds_head(backbone_output)

        stds = torch.exp(log_stds)
        policy = Normal(means, stds)
        action = policy.sample()
        return action, policy

    def get_best_action(self, observation):
        with torch.no_grad():
            backbone_output = self.backbone(observation)
            means = self.means_head(backbone_output)
            return means
