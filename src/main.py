import gin
import torch
import wandb
from torch.optim import Adam

from data.ExperienceBuffer import ExperienceBuffer
from models.Actor import Actor, train_actor
from models.Critic import Critic, train_critic
from utils.utils import create_environment, dict_iter2tensor, set_seed, setup_logger


@gin.configurable()
def main(gamma):
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
    experience_buffer = ExperienceBuffer(gamma)

    # create optimizers
    actor_optimizer = Adam(actor.parameters(), lr=3e-3, weight_decay=10e-4)
    critic_optimizer = Adam(critic.parameters(), lr=3e-3, weight_decay=10e-4)

    for epoch in range(100):
        #   1. simulate data
        for episode in range(20):
            total_reward = 0
            o = env.reset()
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

            print(f"Epoch {epoch}, episode {episode}, total reward {total_reward}")
            wandb.log({"Total Reward": total_reward})

        #   2. train models
        data = experience_buffer.get_data()
        data = dict_iter2tensor(data)
        actor_loss, entropy = train_actor(actor, data, actor_optimizer)
        critic_loss = train_critic(critic, data, critic_optimizer)
        experience_buffer.clear()

        #   3. log data
        wandb.log({"Actor Loss": actor_loss, "Critic Loss": critic_loss, "Entropy": entropy})


if __name__ == "__main__":
    gin.parse_config_file("experiments/dev_config.gin")
    main()
