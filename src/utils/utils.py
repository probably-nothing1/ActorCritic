import os
from itertools import chain, zip_longest

import gin
import gym
import numpy as np
import torch
import torch.nn as nn
import wandb
from gym import wrappers
from torch.nn import Identity, Linear, Tanh


def get_video_filepath(env):
    assert isinstance(env, wrappers.Monitor)

    def pad_episode_number(id):
        return str(1000000 + id)[1:]

    episode_number = pad_episode_number(env.episode_id - 2)
    filepath = f"{env.file_prefix}.video.{env.file_infix}.video{episode_number}.mp4"
    return os.path.join(env.directory, filepath)


class EvalMonitor(wrappers.Monitor):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.mode = "training"

    def _set_mode(self, mode):
        assert mode in ["evaluation", "training"], "Invalid mode"
        self.mode = mode
        self.stats_recorder.type = "t" if mode == "training" else "e"

    def _video_enabled(self):
        return self.video_callable(self.episode_id, self.mode)


@gin.configurable
def create_environment(name, gym_make_kwargs=dict(), save_videos=False, wrapper_kwargs=dict()):
    env = gym.make(name, **gym_make_kwargs)
    if save_videos:

        def record_lambda(episode_id, mode):
            return mode == "evaluation"

        env = EvalMonitor(env, video_callable=record_lambda, **wrapper_kwargs)
        # env = wrappers.Monitor(env, **wrapper_kwargs)
        # env = wrappers.Monitor(env, "./videos/" + str(time.time()) + "/", force=True, write_upon_reset=True)
    return env


@gin.configurable
def setup_logger(name="run-name", notes="", project="project-name", tags=[], save_code=True, monitor_gym=True):
    wandb.init(**locals())


@gin.configurable
def set_seed(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_fully_connected_network(sizes, output_activation_fn=Identity):
    fc_layers = [Linear(in_size, out_size) for in_size, out_size in zip(sizes[:-1], sizes[1:])]
    activations = [Tanh() for _ in range(len(fc_layers) - 1)] + [output_activation_fn()]
    layers = [x for x in chain(*zip_longest(fc_layers, activations)) if x is not None]
    return nn.Sequential(*layers)


def dict_iter2tensor(dict_of_iterable):
    return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in dict_of_iterable.items()}
