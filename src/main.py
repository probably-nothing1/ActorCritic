import gym
import torch

from data.ExperienceBuffer import ExperienceBuffer
from models.Actor import Actor
from models.Critic import Critic

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

    for epoch in range(10):
        #   1. simulate data
        done = False
        o = env.reset()
        while not done:
            o = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                a = actor(o)
                v = critic(o)
            a, v = a.item(), v.item()
            next_o, reward, done, info = env.step(a)

            experience_buffer.append(o, v, a, reward, done)
            o = next_o

    print(experience_buffer.trajectories[0].observations)
    #   2. train models
    #   3. log data
