import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from dynamic_policy_gradient.algs.algs_utils import TabularSoftmaxPolicy, compute_loss
import math

class SoftmaxPG:
    # Initialize the policy network and optimizer
    def __init__(self, env, device="cpu", action_list = None):
        self.env = env
        self.seed = env.seed
        self.optimal_V = env.evaluate_optimal_V()
        self.device = device
        self.action_list = action_list

        if self.seed is not None:
            torch.manual_seed(self.seed)
    
    def init_policy(self, lr):
        self.policy = TabularSoftmaxPolicy(
            self.env.observation_space.n, self.env.action_space.n, self.device, self.action_list
        )
        self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
        

    def train(
        self,
        eps,
        pg_lr,
        discount_factor,
        gradient_clipping,
        reward_normalization,
        computational_power,
        adaptive
    ):  
        num_iter_per_episode= math.ceil(1 / (1 - discount_factor)) #max-iterations per run in PG
        
        if adaptive: 
                # stepsize for PG is the last in DynPG (Horizon is 6)
                pg_lr = 2*(1-self.env.discount_factor) / (1-self.env.discount_factor**6)
        
        # initialize stationary policy
        self.init_policy(pg_lr)

        torch.manual_seed(self.seed)

        logging.info("--------------Start Training--------------")
        episode_rewards = []  # To store the total reward per episode
        episode_V_gaps = []  # To store the V gap per episode
        episode_actual_Vs = []  # To store the achieved V per episode
        sample_path = []
        total_samples = 0
        total_episodes = 0
        V_gap = float("inf")

        # Training loop of stationary policy
        while V_gap > eps and total_samples<computational_power:
            state, _ = self.env.reset()
            total_episodes += 1

            log_probs = []
            rewards = []
            episode_reward = 0
            for _ in range(num_iter_per_episode):
                action, log_prob, _ = self.policy.get_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_samples += 1  # count the number of samples (interactions)
                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)

            #update policy
            self.optimizer.zero_grad()
            loss = compute_loss(
                log_probs, rewards, discount_factor, reward_normalization, self.device
            )
            loss.backward()

            # Clip gradients to prevent explosion
            if gradient_clipping:
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            self.optimizer.step()

            # only for recording purpose
            optimal_V, actual_V = self.env.evaluate_stationary(self)
            V_gap = optimal_V - actual_V
            episode_V_gaps.append(V_gap)
            episode_actual_Vs.append(actual_V)
            sample_path.append(total_samples)

            if total_episodes % 100 ==0:
                # logging and minitoring
                logging.info(
                    f"Samples {total_samples}: Loss = {loss.item():.3f}, V Gap = {V_gap:.3f}"
                )

        # house keeping
        self.episode_rewards = episode_rewards
        self.episode_V_gaps = episode_V_gaps
        self.episode_actual_Vs = episode_actual_Vs
        self.total_episodes = total_episodes
        self.total_samples = total_samples
        self.avg_samples_per_episode = total_samples / total_episodes
        self.sample_path = sample_path
        logging.info(
            f"Total episodes: {total_episodes}, Total samples: {total_samples}."
        )
        logging.info(f"Solution returned with V gap: {V_gap:.3f}")
        logging.info("--------------Training Completed--------------")
        self.env.close()

        
