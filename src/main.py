import gin
import numpy as np
import torch
import wandb
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from models.Actor import Actor, train_actor
from models.Critic import Critic, train_critic
from utils.utils import create_environment, dict_iter2tensor, set_seed, setup_logger


@gin.configurable
def collect_trajectories(actor, critic, env, experience_buffer, min_num_of_steps_in_epoch):
    steps_collected = 0
    while steps_collected < min_num_of_steps_in_epoch:
        o = env.reset()
        total_reward = 0
        done = False
        while not done:
            o = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                a, _ = actor(o)
                v = critic(o)

            a, v = a.item(), v.item()
            next_o, reward, done, info = env.step(a)
            total_reward += reward

            experience_buffer.append(o, v, a, reward, done)
            o = next_o
            steps_collected += 1

        print(f"Total Reward {total_reward}")
        wandb.log({"Total Reward": total_reward})

    print(f"steps collected {steps_collected}")


@gin.configurable
def evaluate(actor, env, runs=20):
    total_rewards = np.zeros(runs)
    for i in range(runs):
        total_reward = 0
        o = env.reset()
        done = False
        while not done:
            o = torch.as_tensor(o, dtype=torch.float32)

            best_action = actor.get_best_action(o)
            next_o, reward, done, info = env.step(best_action.item())

            total_reward += reward
            o = next_o

        total_rewards[i] = total_reward

    return total_rewards.mean(), total_rewards.max(), total_rewards.min()


@gin.configurable
def main(actor_lr, critic_lr, weight_decay, epochs):
    setup_logger()
    set_seed()

    # create env
    env = create_environment()
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # create models
    actor = Actor(observation_dim, action_dim)
    critic = Critic(observation_dim)

    # create exp buffer
    experience_buffer = ExperienceBuffer()

    # create optimizers
    actor_optimizer = Adam(actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
    critic_optimizer = Adam(critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        collect_trajectories(actor, critic, env, experience_buffer)

        data = experience_buffer.get_data()
        data = dict_iter2tensor(data)
        actor_loss, entropy = train_actor(actor, data, actor_optimizer)
        critic_loss = train_critic(critic, data, critic_optimizer)
        experience_buffer.clear()

        test_mean_reward, test_max_reward, test_min_reward = evaluate(actor, env)

        wandb.log(
            {
                "Actor Loss": actor_loss,
                "Critic Loss": critic_loss,
                "Entropy": entropy,
                "Test Mean Reward": test_mean_reward,
                "Test Max Reward": test_max_reward,
                "Test Min Reward": test_min_reward,
            }
        )


if __name__ == "__main__":
    gin.parse_config_file("experiments/dev_config.gin")
    main()
