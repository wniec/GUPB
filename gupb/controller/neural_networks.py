import torch
import torch.nn as nn


class ActorCriticNet(nn.Module):
    def __init__(self, action_size, input_size=140, hidden1=560, hidden2=280):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.LayerNorm(hidden2),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden2, action_size), nn.Softmax(dim=-1)
        )
        self.critic_head = nn.Linear(hidden2, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        policy = self.actor_head(shared_out)
        value = self.critic_head(shared_out)
        return policy, value


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reward, expected_reward, log_prob):
        error = reward - expected_reward
        return -error * log_prob
