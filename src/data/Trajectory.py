import gin
import numpy as np
from scipy.signal import lfilter


@gin.configurable
def advantage_GAE(trajectory, GAE_lambda=0.9):
    residuals = trajectory.gamma * np.append(trajectory.values[1:], 0) - trajectory.values
    deltas = trajectory.rewards + residuals
    deltas_backward = deltas[::-1]
    advantage_GAE_backwards = lfilter([1], [1, -trajectory.gamma * GAE_lambda], deltas_backward)
    return advantage_GAE_backwards[::-1]


@gin.configurable
def total_reward(trajectory):
    total_reward = trajectory.rewards.sum()
    return np.ones(trajectory.rewards.shape) * total_reward


@gin.configurable
def reward_following(trajectory):
    rewards_backward = trajectory.rewards[::-1]
    cumsum_returns_backwards = lfilter([1], [1, -1], rewards_backward, axis=0)
    cumsum_returns = cumsum_returns_backwards[::-1]
    return cumsum_returns


@gin.configurable
def td_residual(trajectory):
    residuals = trajectory.gamma * np.append(trajectory.values[1:], 0) - trajectory.values
    return trajectory.rewards + residuals


@gin.configurable
class Trajectory:
    def __init__(self, gamma, theta_function):
        self.gamma = gamma
        self.length = 0
        self.observations = np.zeros(10000)
        self.values = np.zeros(10000)
        self.actions = np.zeros(10000)
        self.rewards = np.zeros(10000)
        self.discounted_rewards = np.zeros(10000)
        self.theta_function = theta_function
        self.finished = False

    def append(self, observation, value, action, reward):
        assert not self.finished
        if self.length == 0:
            self.observations = np.zeros((10000, *observation.shape))
            if not isinstance(action, int):
                self.actions = np.zeros((10000, *action.shape))

        self.observations[self.length] = observation
        self.values[self.length] = value
        self.actions[self.length] = action
        self.rewards[self.length] = reward
        self.length += 1

    def finish(self):
        self.finished = True
        self.observations = self.observations[: self.length]
        self.values = self.values[: self.length]
        self.actions = self.actions[: self.length]
        self.rewards = self.rewards[: self.length]
        self._compute_discounted_rewards()
        self.theta = self.theta_function(self)
        if len(self.actions.shape) != 1:
            self.theta = np.expand_dims(self.theta, axis=-1)

    def _compute_discounted_rewards(self):
        rewards_backward = self.rewards[::-1]
        discounted_returns_backwards = lfilter([1], [1, -self.gamma], rewards_backward, axis=0)
        self.discounted_rewards = discounted_returns_backwards[::-1]

    def __len__(self):
        return self.length
