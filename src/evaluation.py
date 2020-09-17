import gin
import gym
import numpy as np
import torch
import wandb

from utils.env_utils import get_video_filepath


def record_evaluation_video(actor_critic, env, device):
    is_recording = isinstance(env, gym.wrappers.Monitor)
    if is_recording:
        env._set_mode("evaluation")

    evaluate_one(actor_critic, env, device)

    if is_recording:
        env._set_mode("training")
        env.reset()
        video_filepath = get_video_filepath(env)
        wandb.log({"Evaluate Video": wandb.Video(video_filepath)})


def evaluate_one(actor_critic, env, device):
    total_reward = 0
    o = env.reset()
    done = False
    while not done:
        o = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)

        best_action = actor_critic.get_best_action(o)
        next_o, reward, done, info = env.step(best_action)

        total_reward += reward
        o = next_o

    return total_reward


@gin.configurable
def evaluate(actor_critic, env, device, runs=20):
    total_rewards = np.zeros(runs)
    for i in range(runs):
        total_rewards[i] = evaluate_one(actor_critic, env, device)

    return total_rewards.mean()
