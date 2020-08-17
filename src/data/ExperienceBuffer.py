import numpy as np

from .Trajectory import Trajectory


class ExperienceBuffer:
    def __init__(self, gamma):
        self.trajectories = []
        self.gamma = gamma

    def append(self, observation, value, action, reward, done=False):
        if not self.trajectories or self.trajectories[-1].finished:
            self.trajectories.append(Trajectory(gamma=self.gamma))
        self.trajectories[-1].append(observation, value, action, reward)

        if done:
            self.finish_trajectory()

    def get_data(self):
        return {
            "observations": np.concatenate([t.observations for t in self.trajectories]),
            "values": np.concatenate([t.values for t in self.trajectories]),
            "actions": np.concatenate([t.actions for t in self.trajectories]),
            "rewards": np.concatenate([t.rewards for t in self.trajectories]),
            "discounted_rewards": np.concatenate([t.discounted_rewards for t in self.trajectories]),
            "advantages": np.concatenate([t.advantages for t in self.trajectories]),
            "advantages_GAE": np.concatenate([t.advantages_GAE for t in self.trajectories]),
        }

    def finish_trajectory(self):
        self.trajectories[-1].finish()

    def clear(self):
        self.trajectories.clear()
