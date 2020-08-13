import numpy as np
from scipy.signal import lfilter


class Trajectory:
    def __init__(self, gamma):
        self.gamma = gamma
        self.length = 0
        self.observations = np.zeros(1000)
        self.values = np.zeros(1000)
        self.actions = np.zeros(1000)
        self.rewards = np.zeros(1000)
        self.discounted_rewards = np.zeros(1000)
        self.advantages = np.zeros(1000)
        self.finished = False

    def append(self, observation, value, action, reward):
        assert not self.finished
        self.observations[self.length] = observation
        self.values[self.length] = value
        self.actions[self.length] = action
        self.rewards[self.length] = reward
        self.length = self.length + 1

    def finish(self):
        self.finished = True
        self.observations = self.observations[: self.length]
        self.values = self.values[: self.length]
        self.actions = self.actions[: self.length]
        self.rewards = self.rewards[: self.length]
        self._compute_discounted_rewards()
        self._compute_advantage()

    def _compute_discounted_rewards(self):
        rewards_backward = self.rewards[::-1]
        discounted_returns_backwards = lfilter([1], [1, -self.gamma], rewards_backward, axis=0)
        self.discounted_rewards = discounted_returns_backwards[::-1]

    def _compute_advantage(self):
        return [0] * len(self.values)

    def __len__(self):
        return self.length
