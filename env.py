import gym


def get_env(env_id):
    """Return train and val env"""
    env = gym.make(env_id)
    val_env = gym.make(env_id)

    return env, val_env
