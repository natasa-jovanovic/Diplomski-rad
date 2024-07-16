import gymnasium as gym
import torch
from agent import SACAgent
from replay_buffer import ReplayBuffer
from config import Config
import wandb
import metaworld
import random

def main(config=None):

    with wandb.init(config=config):

        cfg = Config()

        ml1 = metaworld.ML1(cfg.env_name)
        testing_envs = []
        for name, env_cls in ml1.train_classes.items():
            env = env_cls()  # Create an environment
            task = random.choice([task for task in ml1.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            testing_envs.append(env)

        replay_buffer = ReplayBuffer(cfg.replay_buffer_size)
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg)
        config = wandb.config

        cfg.hidden_dim = config.hidden_dim
        cfg.gamma = config.gamma
        cfg.learning_rate = config.learning_rate
        cfg.batch_size = config.batch_size


        for episode in range(cfg.max_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for t in range(cfg.max_timesteps):
                action = agent.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                agent.update(replay_buffer)

                state = next_state
                episode_reward += reward

                if done:
                    break
            wandb.log({"Reward": episode_reward})

if __name__ == "__main__":

    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'Reward',
        'goal': 'maximize'   
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'hidden_dim': {
            'values': [128, 256, 512]
        },
        'gamma': {
            'values': [0.99, 0.995]
        }
    }

    parameters_dict.update({
        'learning_rate': {
            'distribution': 'uniform',
            'min': 1e-4,
            'max': 1e-3
        },
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 32,
            'max': 256,
        }
    })
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="sac-pytorch")
    wandb.agent(sweep_id, function=main)

    #main()
