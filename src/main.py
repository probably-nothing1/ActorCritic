import gin
import torch
import wandb
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from evaluation import evaluate, record_evaluation_video
from models.Actor import create_actor, train_actor
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

            a, v = a.numpy(), v.item()
            next_o, reward, done, info = env.step(a)
            total_reward += reward

            experience_buffer.append(o, v, a, reward, done)
            o = next_o
            steps_collected += 1

        print(f"Total Reward {total_reward}")
        wandb.log({"Total Reward": total_reward})

    print(f"steps collected {steps_collected}")


@gin.configurable
def main(actor_lr, critic_lr, weight_decay, epochs, record_eval_video_rate):
    setup_logger()
    set_seed()

    # create env
    env = create_environment()
    observation_dim = env.observation_space.shape[0]

    # create models
    actor = create_actor(env.observation_space, env.action_space)
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

        if epoch % record_eval_video_rate == 0:
            record_evaluation_video(actor, env)

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
