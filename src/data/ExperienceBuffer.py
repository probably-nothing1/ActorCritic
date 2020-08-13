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

    def finish_trajectory(self):
        self.trajectories[-1].finish()

    def clear(self):
        self.trajectories.clear()
