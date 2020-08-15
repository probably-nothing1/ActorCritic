import gin
import torch.nn as nn

from utils.utils import create_fully_connected_network


@gin.configurable
def train_critic(critic, data, optimizer, num_train_loop_iterations=80):
    optimizer.zero_grad()
    observations = data["observations"]
    discounted_rewards = data["discounted_rewards"]

    for _ in range(num_train_loop_iterations):
        values = critic(observations)
        loss = ((values - discounted_rewards) ** 2).mean()
        loss.backward()
        optimizer.step()

    return loss.item()


@gin.configurable
class Critic(nn.Module):
    def __init__(self, observation_dim, hidden_sizes):
        super().__init__()
        sizes = [observation_dim, *hidden_sizes, 1]
        self.fc_net = create_fully_connected_network(sizes)

    def forward(self, observation):
        logits = self.fc_net(observation)
        return logits
