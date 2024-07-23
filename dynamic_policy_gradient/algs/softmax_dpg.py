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
        #episode_std = []  # To store the std of the achieved V per episode
        sample_path = []
        total_samples = 0
        total_episodes = 0
        h = 0 # iteration index of dpg
        V_gap = float("inf")
        #cost_to_go_h = torch.zeros(self.env.observation_space.n, device=self.device, requires_grad=False)

        H = math.ceil(torch.log(torch.tensor((1-self.env.discount_factor) *eps/20,dtype=float)) / torch.log(torch.tensor(self.env.discount_factor,dtype=float)))
        #logging.info(f"Horizon is: {H}")

        constant = 0
        for t in range(H):
            constant += ((1- (self.env.discount_factor**(t+1))) *self.env.discount_factor**(H-t-1))**0.5
        #constant = (eps/2)* (1/constant)
        #logging.info(f"constant is: {constant}")
        
        while V_gap > eps and total_samples<computational_power:
            if adaptive: 
                lr_h = 2*(1-self.env.discount_factor) / (1-(self.env.discount_factor**(h+1)))
                episode_h = math.ceil(50* (1-(self.env.discount_factor**(h+1))) / (1-self.env.discount_factor)) # number of episodes for the current horizon
            else:
                lr_h = lr[h] # learning rate for the current horizon
                episode_h = Ns[h] # number of episodes for the current horizon

            

            self.init_policy(h, copy_weights, lr_h)
            #scheduler = optim.lr_scheduler.StepLR(self.optimizers[0], step_size=30, gamma=0.9)

            # num_iter_per_episode starts from 1 to infinity (horizon of the current training epoch)
            num_iter_per_episode = h + 1
            #logging.info(
            #    f"h={h}, training MDP with {num_iter_per_episode} iterations per episode (horizon)"
            #)

            V_h_gap = float("inf")

            #eps_h = 0.00001*((eps*0.5)/constant) * ((1-(self.env.discount_factor**(h+1)))**0.5/(self.env.discount_factor**(H-h-1) )**(0.5))
            #logging.info(f"eps_h is: {eps_h}")
            actual_V_h_old = 100

            for _ in range(episode_h):
                

                # random init
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
                #scheduler.step()

                # evaluate the policy over finite horizon
                optimal_V_h, actual_V_h = self.env.evaluate_h(self,num_iter_per_episode)
                V_h_gap = np.abs(actual_V_h_old - actual_V_h)
                #logging.info(f"actual_value is: {actual_V_h} and optimal_value is: {optimal_V_h}")
                actual_V_h_old = actual_V_h
                
                # only for recording purpose
                optimal_V, actual_V  = self.env.evaluate_stationary(self)
                episode_V_gaps.append(optimal_V - actual_V)
                episode_actual_Vs.append(actual_V)
                sample_path.append(total_samples)

                #if total_episodes %100:
                #    logging.info(f"V gap is : {(optimal_V - actual_V):.3f}")
                
                if total_samples>computational_power:
                    break
                
                # stop training if the h-gap is small enough
                #if V_h_gap < 0.5**(h+1):
                #    logging.info(
                #    f"Training of epoch {h} stopped early because V_h gap is small enough: {V_h_gap:.3f}"
                #    )
                #    break

                # stop training if the overall gap is small enough
                if optimal_V - actual_V < eps:
                    logging.info(
                    f"Training of epoch {h} stopped early because V gap is small enough: {(optimal_V - actual_V):.3f}"
                    )
                    break

                #logging.info(
                #    f"Episode {total_episodes}: Optimal V = {optimal_V:.3f}, Actual V = {actual_V:.3f}, Optimal V_h = {optimal_V_h:.3f}, Actual V_h = {actual_V_h:.3f}, V_h Gap = {V_h_gap:.3f}, Lr = {self.optimizers[0].param_groups[0]['lr']:.3f}"
                #)


            #logging.info(f"Policy is: \n {F.softmax(self.policies[0].logits, dim=1).detach()}")
            #logging.info(
                #f"Finished iteration h = {h}, V Gap = {optimal_V - actual_V:.3f}, Actual V = {actual_V:.3f}"
            #)
            V_gap = optimal_V - actual_V
            h += 1

        # house keeping
        self.episode_rewards = episode_rewards
        self.episode_V_gaps = episode_V_gaps
        self.episode_actual_Vs = episode_actual_Vs
        #self.episode_std = episode_std
        self.total_episodes = total_episodes
        self.total_samples = total_samples
        self.avg_samples_per_episode = total_samples / total_episodes
        self.sample_path = sample_path
        logging.info(
            f"Total episodes: {total_episodes}, Total samples: {total_samples}, Avg Samples per episode: {total_samples/total_episodes:.3f}"
        )
        logging.info(f"Solution returned with V gap: {V_gap:.3f}")
        logging.info("--------------Training Completed--------------")
        self.env.close()

        
