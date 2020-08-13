from pytest import fixture

from ..src.data.Trajectory import Trajectory


@fixture(scope="function")
def empty_trajectory(request):
    yield Trajectory(request.param)


@fixture(scope="function")
def zero_rewards_trajectory(request):
    trajectory = Trajectory(request.param)
    for i in range(10):
        trajectory.append(0, 0, 0, 0)
    yield trajectory


@fixture(scope="function")
def ones_rewards_trajectory(request):
    trajectory = Trajectory(request.param)
    for i in range(3):
        trajectory.append(0, 0, 0, 1)
    yield trajectory
