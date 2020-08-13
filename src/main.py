import gym
import torch
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from models.Actor import Actor
from models.Critic import Critic
from utils.pytorch_utils import dict_iter2tensor


def train_actor(actor, data, optimizer):
    optimizer.zero_grad()
    observations = data["observations"]
    actions = data["actions"]
    discounted_rewards = data["discounted_rewards"]
    _, policy = actor(observations)
    log_probs = policy.log_prob(actions)
    loss = -(log_probs * discounted_rewards).mean()

    loss.backward()
    optimizer.step()


def train_critic(critic, data, optimizer):
    optimizer.zero_grad()
    observations = data["observations"]
    discounted_rewards = data["discounted_rewards"]
    values = critic(observations)
    loss = ((values - discounted_rewards) ** 2).mean()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # create wandb

    # create env
    env = gym.make("CartPole-v0")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # create models
    actor = Actor(observation_dim, action_dim)
    critic = Critic(observation_dim)

    # create exp buffer
    experience_buffer = ExperienceBuffer(gamma=0.99)

    # create optimizers and train functions
    actor_optimizer = Adam(actor.parameters(), lr=3e-3, weight_decay=10e-4)
    critic_optimizer = Adam(critic.parameters(), lr=3e-3, weight_decay=10e-4)

    for epoch in range(100):
        #   1. simulate data
        for episode in range(20):
            total_reward = 0
            o = env.reset()
            done = False
            while not done:
                o = torch.as_tensor(o, dtype=torch.float32)
                with torch.no_grad():
                    a, _ = actor(o)
                    v = critic(o)
                a, v = a.item(), v.item()
                next_o, reward, done, info = env.step(a)
                total_reward += reward

                experience_buffer.append(o, v, a, reward, done)
                o = next_o
            print(f"Epoch {epoch}, episode {episode}, total reward {total_reward}")

        #   2. train models
        data = experience_buffer.get_data()
        data = dict_iter2tensor(data)
        train_actor(actor, data, actor_optimizer)
        train_critic(critic, data, critic_optimizer)
        experience_buffer.clear()

        #   3. log data
