import torch

class Config:
    env_name = "door-lock-v2"
    replay_buffer_size = 100000000
    max_episodes = 10000
    max_timesteps = 2000
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    lr = 3e-4
    hidden_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = 0.2  # Entropy regularization coefficient

    def __init__(self):
        pass

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}