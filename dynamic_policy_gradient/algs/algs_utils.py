import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class TabularSoftmaxPolicy(nn.Module):
    def __init__(self, num_states, num_actions, device, action_list = None):
        super(TabularSoftmaxPolicy, self).__init__()
        if action_list is None:
            logits = torch.zeros((num_states, num_actions), device=device)
        else:
            logits = torch.zeros((num_states, num_actions), device=device)
            # action masking
            for state in range(num_states):
                logits[state, action_list[state]:] = -float("inf")
        self.logits = nn.Parameter(logits)
     

    def forward(self, state):
        return F.softmax(self.logits[state], dim=-1)

    def get_action(self, state):
        probs = self.forward(state)
        try:
            dist = torch.distributions.Categorical(probs)
        except:
            return 1, 1, 1
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, probs


def compute_loss(log_probs, rewards, discount_factor, reward_normalization, device):
    discounted_rewards = []
    Gt = 0
    for reward in rewards[::-1]:
        Gt = reward + discount_factor * Gt
        discounted_rewards.insert(0, Gt)

    discounted_rewards = torch.tensor(
        discounted_rewards, dtype=torch.float32, device=device
    )
    if reward_normalization and len(discounted_rewards) > 1:
       discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
           discounted_rewards.std() + 1e-9
       )  # Normalize

    policy_loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
    return policy_loss

def compute_loss_h(log_probs, rewards, discount_factor, reward_normalization, device):
    discounted_rewards = []
    Gt = 0
    for reward in rewards[::-1]:
        Gt = reward + discount_factor * Gt
        discounted_rewards.insert(0, Gt)

    discounted_rewards = torch.tensor(
        discounted_rewards, dtype=torch.float32, device=device
    )
    if reward_normalization and len(discounted_rewards) > 1:
       discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
           discounted_rewards.std() + 1e-9
       )  # Normalize

    policy_loss = -((log_probs[0]) * discounted_rewards[0])
    return policy_loss