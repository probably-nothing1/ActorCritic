import gin
import gym
import numpy as np
import torch
import torch.nn as nn
import wandb
from gym import wrappers


@gin.configurable
def create_environment(name, gym_make_kwargs=dict(), save_videos=False, wrapper_kwargs=dict()):
    env = gym.make(name, **gym_make_kwargs)
    if save_videos:
        env = wrappers.Monitor(env, **wrapper_kwargs)
    # env = wrappers.Monitor(env, "./videos/" + str(time.time()) + "/", force=True, write_upon_reset=True)
    return env


@gin.configurable
def setup_logger(name="run-name", notes="", project="project-name", tags=[], save_code=True, monitor_gym=True):
    wandb.init(**locals())


@gin.configurable
def set_seed(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def dict_iter2tensor(dict_of_iterable):
    return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in dict_of_iterable.items()}
