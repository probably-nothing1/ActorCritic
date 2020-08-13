from pytest import mark


@mark.trajectory
class TrajectoryTests:
    @mark.parametrize("empty_trajectory", [(1)], indirect=["empty_trajectory"])
    def test_append_to_empty_trajectory(self, empty_trajectory):
        observation, value, action, reward = 0, 0, 0, 0
        empty_trajectory.append(observation, value, action, reward)
        assert len(empty_trajectory) == 1

    @mark.parametrize("ones_rewards_trajectory", [(1)], indirect=["ones_rewards_trajectory"])
    def test_append_to_non_empty_trajectory(self, ones_rewards_trajectory):
        ones_rewards_trajectory_len = len(ones_rewards_trajectory)
        observation, value, action, reward = 0, 0, 0, 0
        ones_rewards_trajectory.append(observation, value, action, reward)
        assert len(ones_rewards_trajectory) == ones_rewards_trajectory_len + 1

    @mark.parametrize("ones_rewards_trajectory", [(0.99)], indirect=["ones_rewards_trajectory"])
    def test_compute_discounted_rewards(self, ones_rewards_trajectory):
        ones_rewards_trajectory.finish()
        discounted_rewards = ones_rewards_trajectory.discounted_rewards
        rewards = ones_rewards_trajectory.rewards
        dr2 = rewards[2]
        dr1 = dr2 + 0.99 * rewards[1]
        dr0 = dr1 + 0.99 * rewards[0]
        for x, y in zip(discounted_rewards, [dr0, dr1, dr2]):
            assert 0.995 * y < x < y * 1.005

    # @mark.parametrize("ones_rewards_trajectory", [(0.99)], indirect=["ones_rewards_trajectory"])
    # def test_compute_advantage(self, ones_rewards_trajectory):
    #     ones_rewards_trajectory.finish()
    #     discounted_rewards = ones_rewards_trajectory.discounted_rewards
    #     rewards = ones_rewards_trajectory.rewards
    #     print(discounted_rewards)
    #     print(rewards)
    #     dr2 = rewards[2]
    #     dr1 = dr2 + 0.99 * rewards[1]
    #     dr0 = dr1 + 0.99 * rewards[0]
    #     for x, y in zip(discounted_rewards, [dr0, dr1, dr2]):
    #         assert 0.995 * y < x < y * 1.005
