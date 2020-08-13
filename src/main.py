import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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


if __name__ == "__main__":
    # create wandb
    # create env
    env = gym.make("CartPole-v0")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor = Actor(observation_dim, action_dim)
    critic = Critic(observation_dim)
    # create exp buffer
    # create optimizers and train functions

    for epoch in range(10):
        #   1. simulate data
        done = False
        o = env.reset()
        while not done:
            with torch.no_grad():
                # o = torch.from_numpy(o)
                o = torch.as_tensor(o, dtype=torch.float32)
                a = actor(o)
                v = critic(o)
                a, v = a.numpy(), v.numpy()
                print(f"Picked action {a}. Value of state: {v}")
                next_o, reward, done, info = env.step(a)
                print(f"Next obs {next_o}\n reward {reward}\n done {done}\n info {info}\n")
                o = next_o
        #   2. train models
        #   3. log data
