import gin

from models.ActorCritic import ActorCritic, AtariActorCritic


@gin.configurable
def create_dqn_agent(env):
    env_name = env.unwrapped.spec.id
    # action_space = env.action_space
    # observation_shape = env.observation_space.shape
    if env_name in ["CartPole-v0"]:
        return ActorCritic(env)
    elif env_name.startswith("Pong"):
        return AtariActorCritic(env)

    raise ValueError(f"Can't create DQN model for {env_name} environment")
