import gin
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.utils import create_fully_connected_network

# def train_actor(actor, data, optimizer):
#     optimizer.zero_grad()
#     observations = data["observations"]
#     actions = data["actions"]
#     theta = data["theta"]

#     _, policy = actor(observations)
#     log_probs = policy.log_prob(actions)
#     loss = -(theta * log_probs).mean()

#     entropy = policy.entropy().mean()

#     loss.backward()
#     optimizer.step()
#     return loss.item(), entropy.item()


@gin.configurable
def create_actor(input_dim, action_space, hidden_sizes):
    if isinstance(action_space, Discrete):
        action_dim = action_space.n
        return DiscreteActor(input_dim, action_dim, hidden_sizes)
    elif isinstance(action_space, Box):
        action_dim = action_space.shape[0]
        return ContinuousActor(input_dim, action_dim, hidden_sizes)
    else:
        raise ValueError(f"Unrecognized Action Space type: {type(action_space)}")


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes):
        super().__init__()
        sizes = [input_dim, *hidden_sizes, action_dim]
        self.fc_net = create_fully_connected_network(sizes)

    def forward(self, observation):
        logits = self.fc_net(observation)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return action, policy

    @torch.no_grad()
    def get_best_action(self, observation):
        logits = self.fc_net(observation)
        return torch.argmax(logits)


class ContinuousActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes):
        super().__init__()
        backbone_sizes = [input_dim, *hidden_sizes[:-1]]
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

    @torch.no_grad()
    def get_best_action(self, observation):
        backbone_output = self.backbone(observation)
        means = self.means_head(backbone_output)
        return means
