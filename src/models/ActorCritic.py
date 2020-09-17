import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Actor import create_actor
from models.Critic import Critic
from utils.utils import create_conv_network


@gin.configurable
def train_actor(actor_critic, data, optimizer):
    optimizer.zero_grad()
    observations = data["observations"]
    actions = data["actions"]
    theta = data["theta"]

    _, policy = actor_critic.actor_forward(observations)
    log_probs = policy.log_prob(actions)
    loss = -(theta * log_probs).mean()

    entropy = policy.entropy().mean()

    loss.backward()
    optimizer.step()
    return loss.item(), entropy.item()


@gin.configurable
def train_critic(actor_critic, data, optimizer, num_train_loop_iterations=80):
    optimizer.zero_grad()
    observations = data["observations"]
    discounted_rewards = data["discounted_rewards"]

    for _ in range(num_train_loop_iterations):
        values = actor_critic.critic_forward(observations)
        loss = F.mse_loss(values, discounted_rewards)
        loss.backward()
        optimizer.step()

    return loss.item()


@gin.configurable
class AtariActorCritic(nn.Module):
    def __init__(self, env, conv_sizes, fc_sizes, use_bn=False):
        super().__init__()
        conv_sizes[0][0] = env.observation_space.shape[0]
        conv_net = create_conv_network(conv_sizes, use_bn=use_bn)
        self.conv_net = nn.Sequential(conv_net, nn.Flatten())
        fc_input_size = self._compute_conv_out(self.conv_net, env.observation_space.shape)

        self.actor = create_actor(fc_input_size, env.action_space, fc_sizes)
        self.critic = Critic(fc_input_size, fc_sizes)

    @torch.no_grad()
    def _compute_conv_out(self, conv_net, observation_shape):
        dummy_input = torch.zeros(1, *observation_shape)
        out = conv_net(dummy_input)
        return out.shape[-1]

    @torch.no_grad()
    def forward(self, observations):
        conv_out = self.conv_net(observations)
        action, policy = self.actor(conv_out)
        value = self.critic(conv_out)
        return action.item(), value.item(), policy

    @torch.no_grad()
    def get_best_action(self, observation):
        conv_out = self.conv_net(observation)
        return self.actor.get_best_action(conv_out).item()

    def actor_forward(self, observations):
        conv_out = self.conv_net(observations)
        return self.actor(conv_out)

    def critic_forward(self, observations):
        conv_out = self.conv_net(observations)
        values = self.critic(conv_out)
        return values.squeeze(-1)


@gin.configurable
class ActorCritic(nn.Module):
    def __init__(self, env, hidden_sizes):
        super().__init__()
        fc_input_size = env.observation_space.shape[0]

        self.actor = create_actor(fc_input_size, env.action_space, hidden_sizes)
        self.critic = Critic(fc_input_size, hidden_sizes)

    @torch.no_grad()
    def forward(self, observations):
        action, policy = self.actor(observations)
        value = self.critic(observations)
        return action.item(), value.item(), policy

    @torch.no_grad()
    def get_best_action(self, observation):
        return self.actor.get_best_action(observation).item()

    def actor_forward(self, observations):
        return self.actor(observations)

    def critic_forward(self, observations):
        values = self.critic(observations)
        return values.squeeze(-1)
