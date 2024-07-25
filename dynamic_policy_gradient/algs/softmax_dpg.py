import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from dynamic_policy_gradient.algs.algs_utils import TabularSoftmaxPolicy, compute_loss_h
#from dynamic_policy_gradient.utils.hallway import evaluate_stationary, evaluate_h, evaluate_optimal_V
import math
import numpy as np

class DynamicSoftmaxPG:
    # Initialize the policy network and optimizer
    def __init__(self, env, device="cpu", action_list = None):
        self.env = env
        self.optimal_V = env.evaluate_optimal_V()
        #logging.info(f"Optimal_value is: {self.optimal_V}")
        self.device = device
        self.action_list = action_list

        # Initialize the policy network and optimizer
        self.policies = []
        self.optimizers = []

        self.seed = env.seed
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            #logging.info(f"Random seed set to: {self.seed}")

    def init_policy(self, h, copy_weights, lr):
        policy = TabularSoftmaxPolicy(
            self.env.observation_space.n,
            self.env.action_space.n,
            self.device,
            self.action_list
        )

        # Copy logits from the previous iteration if exists
        if copy_weights and h > 0:
            with torch.no_grad():
                policy.logits.data = self.policies[0].logits.data.clone()

        optimizer = optim.SGD(policy.parameters(), lr=lr)

        self.policies.insert(0, policy)
        self.optimizers.insert(0, optimizer)

    def train(
        self,
        eps,
        lr,
        adaptive,
        Ns,
        discount_factor,
        gradient_clipping,
        reward_normalization,
        copy_weights,
        computational_power,
    ):
        torch.manual_seed(self.seed)

        logging.info("--------------Start Training--------------")

        # house keeping
        episode_rewards = []
        episode_V_gaps = []  # To store the V gap per episode
        episode_actual_Vs = []  # To store the achieved V per episode
        sample_path = []
        total_samples = 0
        total_episodes = 0
        h = 0 # iteration index of dpg
        V_gap = float("inf")

        # Outer training loop over horizons in DynPG 
        while V_gap > eps and total_samples<computational_power:
            if adaptive: 
                lr_h = 2*(1-self.env.discount_factor) / (1-(self.env.discount_factor**(h+1)))
                N_h = math.ceil(45* (1-(self.env.discount_factor**(h+1))) / (1-self.env.discount_factor)) # number of episodes for the current horizon
            else:
                try:
                    lr_h = lr[h] # learning rate for the current horizon
                    N_h = Ns[h] # number of episodes for the current horizon
                except:
                    logging.info("There are no more prefixed learning rates and number of episodes to continue training")
                    break
            logging.info(f"Training of epoch {h} started with lr: {lr_h:.3f} and {N_h} episodes")

            # add policy pi_h in the beginning of the list
            self.init_policy(h, copy_weights, lr_h)

            # num_iter_per_episode starts from 1 to infinity (determinsitic horizon for training)
            num_iter_per_episode = h + 1

            # Inner training loop for policy pi_h
            for _ in range(N_h):
                state, _ = self.env.reset()
                total_episodes += 1

                log_probs = []
                rewards = []
                episode_reward = 0
                for step in range(num_iter_per_episode):
                    action, log_prob, _ = self.policies[step].get_action(state)
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    total_samples += 1  # count the number of samples (interactions)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)

                # update a single policy corresponds to the current horizon
                self.optimizers[0].zero_grad()
                loss = compute_loss_h(
                    log_probs, rewards, discount_factor, reward_normalization, self.device
                )
                loss.backward()

                # Clip gradients to prevent explosion
                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.policies[0].parameters(), max_norm=1.0)

                self.optimizers[0].step()
                
                
                # only for recording purpose
                optimal_V, actual_V  = self.env.evaluate_stationary(self)
                episode_V_gaps.append(optimal_V - actual_V)
                episode_actual_Vs.append(actual_V)
                sample_path.append(total_samples)

                if total_samples>computational_power:
                    break
                

                # stop training if the overall gap is small enough
                if optimal_V - actual_V < eps:
                    logging.info(
                    f"Training of epoch {h} stopped early because V gap is small enough: {(optimal_V - actual_V):.3f}"
                    )
                    break


            logging.info(
                f"Finished iteration h = {h}, V Gap = {optimal_V - actual_V:.3f}, Total samples = {total_samples}"
            )
            V_gap = optimal_V - actual_V
            h += 1

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

        
