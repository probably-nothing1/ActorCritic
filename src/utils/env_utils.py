import os

import gin
import gym
from gym import wrappers


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
