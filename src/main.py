import sys

import gin
import torch
import wandb
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from evaluation import evaluate, record_evaluation_video
from models.ActorCritic import create_actor_critic, train_actor, train_critic
from utils.env_utils import create_environment
from utils.utils import dict_iter2tensor, set_seed, setup_logger


@gin.configurable
def collect_trajectories(actor_critic, env, experience_buffer, min_num_of_steps_in_epoch, device="cpu"):
    steps_collected = 0
    while steps_collected < min_num_of_steps_in_epoch:
        o = env.reset()
        total_reward = 0
        done = False
        while not done:
            o_tensor = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)

            a, v, _ = actor_critic(o_tensor)

            next_o, reward, done, info = env.step(a)
            total_reward += reward

            experience_buffer.append(o, v, a, reward, done)
            o = next_o
            steps_collected += 1

        print(f"Total Reward {total_reward}")
        wandb.log({"Total Reward": total_reward})

    print(f"steps collected {steps_collected}")


@gin.configurable
def main(lr, weight_decay, epochs, record_eval_video_rate, device, solved_score):
    env = create_environment()

    setup_logger()
    set_seed(env)

    actor_critic = create_actor_critic(env).to(device)
    print(actor_critic)

    experience_buffer = ExperienceBuffer()

    optimizer = Adam(actor_critic.parameters(), lr=lr, weight_decay=weight_decay)

    ma_reward = 0
    for epoch in range(epochs):
        collect_trajectories(actor_critic, env, experience_buffer, device=device)

        data = experience_buffer.get_data()
        data = dict_iter2tensor(data, device=device)
        actor_loss, entropy = train_actor(actor_critic, data, optimizer)
        critic_loss = train_critic(actor_critic, data, optimizer)
        experience_buffer.clear()

        test_mean_reward = evaluate(actor_critic, env, device)
        ma_reward = 0.99 * ma_reward + 0.01 * test_mean_reward
        if ma_reward >= solved_score:
            break

        if epoch % record_eval_video_rate == 0:
            record_evaluation_video(actor_critic, env, device)

        wandb.log(
            {
                "Test Reward Moving Average": ma_reward,
                "Epoch": epoch,
                "Actor Loss": actor_loss,
                "Critic Loss": critic_loss,
                "Entropy": entropy,
                "Test Average Reward": test_mean_reward,
            }
        )


if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    main()
