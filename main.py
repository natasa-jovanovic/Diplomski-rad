import Metaworld.metaworld as metaworld
import random

import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
from scripts.sac.sac import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(env_name, i):
    def _init():
        ml1 = metaworld.ML1(env_name)

        env = ml1.train_classes[env_name](render_mode='rgb_array')  # Create an environment with task `pick_place`
        task = random.choice(ml1.train_tasks)
        env.set_task(task)  # Set task
        env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
        env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
        return env
    
    return _init



env_names = ["drawer-open-v2"]
env_fns = [make_env(env_name, i) for i, env_name in enumerate(env_names)]

config = {
    "policy_type": 'MlpPolicy',
    "total_timesteps": 300000,
#    "env_name": [ "door-open-v2", "drawer-open-v2" ]
    "env_name": env_names
}

run = wandb.init(
    config=config,
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="MetaWorld-SAC",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)

env = DummyVecEnv(env_fns)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

run.finish()