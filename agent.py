import torch
import torch.nn.functional as F
import numpy as np
from networks import Actor, Critic
from utils import soft_update

class SACAgent:
    def __init__(self, state_dim, action_dim, config):
        self.actor = Actor(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.critic1 = Critic(state_dim + action_dim, config.hidden_dim).to(config.device)
        self.critic2 = Critic(state_dim + action_dim, config.hidden_dim).to(config.device)
        self.target_critic1 = Critic(state_dim + action_dim, config.hidden_dim).to(config.device)
        self.target_critic2 = Critic(state_dim + action_dim, config.hidden_dim).to(config.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.lr)

        self.replay_buffer = None
        self.device = config.device
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha

        soft_update(self.target_critic1, self.critic1, 1.0)
        soft_update(self.target_critic2, self.critic2, 1.0)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.actor(state).cpu().detach().numpy()

    def update(self, replay_buffer):
        if len(replay_buffer) < replay_buffer.capacity:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample()

        # Critic update
        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        new_actions = self.actor(states)
        actor_loss = -self.critic1(states, new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
