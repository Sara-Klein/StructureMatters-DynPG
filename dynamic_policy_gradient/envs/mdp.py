import torch
import gymnasium
import logging
import numpy as np 

class FiniteMDP(gymnasium.Env):
    # Initialize a MDP environment

    def __init__(self, env_id, n_states, n_actions, discount_factor, episode_length, varying_action_spaces=False, seed=None):
        """
        Initialize the finite MDP object.

        :param n_states: An integer representing the number of states in the MDP.
        :param n_actions: An integer representing the number of actions in the MDP.
        :param discount_factor: A float representing the discount factor for future rewards.
        """
        self.spec = type("spec", (object,), {"id": env_id})
        self.action_space = gymnasium.spaces.Discrete(n=n_actions)
        self.observation_space = gymnasium.spaces.Discrete(n=n_states)
        self.discount_factor = discount_factor
        self.episode_length = episode_length
        self.rng = torch.Generator()
        self.seed = seed
        if seed is not None:
            self.rng.manual_seed(self.seed)

        # Initialize the state transition function as a 3D PyTorch tensor
        # Dimensions: [current_state, action, next_state]
        self.transition_function = torch.zeros(n_states, n_actions, n_states)

        # Initialize the reward function as a 2D PyTorch tensor
        # Dimensions: [state, action]
        self.reward_function = torch.zeros(n_states, n_actions)
        
        # if reward reandom add standard deviation
        self.std_reward_function = torch.zeros(n_states, n_actions)

        # Set the terminal state
        self.terminal_states = []

        # Handle varying action spaces
        if not varying_action_spaces:
            self.available_actions = {state: list(range(n_actions)) for state in range(n_states)}
        else:
            self.available_actions = {state: [] for state in range(n_states)}

        # Initialize the current episode
        self.reset()

    def step(self, action):
        """
        Perform an action in the MDP, transitioning to the next state and returning the reward.

        :param action: The action to be taken in the current state.
        :return: A tuple containing the next state, the reward received, and a flag indicating whether the next state is terminal.
        """

        if not self.is_action_available(self.current_state, action):
            raise ValueError(f"Action {action} is not available in state {self.current_state}")
        # Transition to next state based on probabilities
        probabilities = self.transition_function[self.current_state, action]
        next_state = torch.multinomial(probabilities, 1).item()

        # Get the reward for the current state-action pair (check if random or not)
        if self.std_reward_function[self.current_state, action] != 0:
            reward = torch.normal(self.reward_function[self.current_state, action], self.std_reward_function[self.current_state, action])
        else:
            reward = self.reward_function[self.current_state, action].item()
        # Update the environment
        self.current_state = next_state

        # terminated if we have reached a terminal state, truncated if the maximum episode length has been reached
        terminated = True if self.current_state in self.terminal_states else False
        self.step_count += 1
        truncated = True if self.step_count >= self.episode_length else False

        # return the observation and stage cost
        return next_state, reward, terminated, truncated, {}

    def reset(self, state=None, seed=None):
        """
        Reset the MDP to the initial state and optionally set a new random seed.

        :param seed: Optional. A seed for the random number generator for reproducibility.
        """
        if seed is not None:
            self.rng.manual_seed(seed)
        if state is None:
            # reset the environment to a random state (uniform distribution) except terminal states
            valid_states = [s for s in range(self.observation_space.n) if s not in self.terminal_states]
            self.current_state = valid_states[
                torch.randint(0, len(valid_states), (1,), generator=self.rng).item()
            ]
        else:
            # check whether the state is valid
            assert 0 <= state < self.observation_space.n, f"Invalid state {state}"
            self.current_state = state
        self.step_count = 0

        return self.current_state, {}

    def set_transition(self, state, action, p):
        """
        Set the transition probabilities for a given state and action using PyTorch tensors.

        :param state: The current state.
        :param action: The action taken.
        :param p: A list or tensor of transition probabilities to each state.
        """
        p = torch.tensor(p)
        if p.sum().item() != 1.0 or len(p) != self.observation_space.n:
            raise ValueError(
                "Transition probabilities must sum to 1 and match the number of states."
            )
        self.transition_function[state, action] = p

    def set_reward(self, state, action, reward):
        """
        Set the reward for a given state and action using PyTorch tensors.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The mean reward received.
        """
        self.reward_function[state, action] = reward

    def set_reward_std(self, state, action, std):
        """
        Set the standard deviation of reward for a given state and action using PyTorch tensors.

        :param state: The current state.
        :param action: The action taken.
        :param std: The standard deviation for the normal distributed reward.
        """
        self.std_reward_function[state, action] = std

    def set_available_actions(self, state, action_list):
        """
        Set the available actions for a given state.

        :param state: The state for which to set available actions.
        :param actions: A list of available actions in the given state.
        """
        if max(action_list) >= self.action_space.n:
            raise ValueError("Action dimension exceeds the global maximum.")
        self.available_actions[state] = list(action_list)

    def is_action_available(self, state, action):
        """
        Check if the given action is available in the given state.

        :param state: The state to check the action availability in.
        :param action: The action to check.
        :return: True if the action is available in the state, False otherwise.
        """
        return action in self.available_actions[state]

    def add_terminal_state(self, state):
        """
        Add a terminal state to the MDP.

        :param state: The terminal state to add.
        """
        self.terminal_states.append(state)

    def render(self):
        pass

    def close(self):
        pass
    
    def evaluate_optimal_V(self):
        gamma = self.discount_factor
        if self.spec.id == "Sutton-extended":
            V = [0,0,0,0,0,0,0] 
            init_states =[1,2,4,5]

            # apply Bellman operator until convergence
            eps= 1

            while eps > 0.00001:
                V_new = [0,0,0,0,0,0,0] 
                for s in init_states:
                    match s:
                        case 1: 
                            V_new[s] = torch.max(self.reward_function[s, :]).item()
                        case 2:
                            V_new[s] = max(self.reward_function[s, 0].item() + gamma * V[1], self.reward_function[s, 1].item() + gamma * V[3], self.reward_function[s, 2].item() + gamma * V[4])
                        case 4:
                            V_new[s] = torch.max(self.reward_function[s, :] + gamma * V[5]).item()    
                        case 5:
                            V_new[s] = torch.max(self.reward_function[s, :]).item()
                eps = torch.max(torch.abs(torch.tensor(V) - torch.tensor(V_new))).item()
                V = V_new
                
            V = torch.tensor([V[s] for s in init_states]).squeeze()
            return  torch.mean(V)
    
        else:
            raise ValueError("This function is not implemented for this environment")
    
    def evaluate_stationary(self, alg):
        # returns the optimal value function and the actual value function for the stationary policy
        # using the infinite horizon Bellman operator to evaluate the actual policy 
        # in DynPG the policy that is currently trained is evaluated

        if self.spec.id == "Sutton-extended":
            optimal_V = alg.optimal_V 
            init_states = [1, 2, 4, 5]
            actual_V = np.zeros(7)
            gamma = alg.env.discount_factor
            eps= 1

            while eps > 0.0001:
                actual_V_new = np.zeros(7)
                for i in range(4):
                    state = init_states[i]
                    if hasattr(alg, "policies"):  # dynamic softmax PG
                        _, _, action_probs = alg.policies[0].get_action(state)
                    else:  # vanilla softmax PG
                        _, _, action_probs = alg.policy.get_action(state)
                    
                    match state:
                        case 1: 
                            actual_V_new[state] = torch.dot(action_probs, self.reward_function[state, :]).item()
                        case 2:
                            help2 = action_probs[torch.nonzero(action_probs)].squeeze()
                            actual_V_new[state] = torch.dot(help2, torch.tensor([self.reward_function[state, 0] + gamma * actual_V[1], self.reward_function[state, 1] + gamma * actual_V[3], self.reward_function[state, 2] + gamma * actual_V[4]])).item()
                        case 4:
                            x2 = self.reward_function[state, :] + torch.ones_like(torch.tensor(self.reward_function[state, :]))* gamma * actual_V[5]
                            actual_V_new[state] = torch.dot(action_probs, x2).item() 
                        case 5:
                            actual_V_new[state] = torch.dot(action_probs, self.reward_function[state, :]).item()
                    
                eps = torch.max(torch.abs(torch.tensor(actual_V) - torch.tensor(actual_V_new))).item()
                actual_V = actual_V_new
            # return the average over states
            actual_V = torch.tensor([actual_V[s] for s in init_states]).squeeze()
            return optimal_V, torch.mean(actual_V)
        else:
            raise ValueError("This function is not implemented for this environment")


    def evaluate_h(self, alg, horizon):
        # calculate $V_h$ for the finite horizon policy that is trained so far 
        # first: use backward induction to calculate the value function for the finite horizon policy
        # second: calculate the best possible optimal_V_h given the fixed future policies
        
        if self.spec.id == "Sutton-extended":
            gamma = self.discount_factor
            init_states = [1, 2,  4, 5]
            actual_V_h = np.zeros(7)
            optimal_V_h_old = np.zeros(7)
            optimal_V_h = np.zeros(7)

            # caluclate the value function for the finite horizon policy by backward induction
            for h in range(horizon):
                actual_V_new = np.zeros(7)
                for i in range(4):
                    state = init_states[i]
                    if hasattr(alg, "policies"):  # dynamic softmax PG
                        _, _, action_probs = alg.policies[(horizon-1-h)].get_action(state)
                    else:  # vanilla softmax PG
                        _, _, action_probs = alg.policy.get_action(state)
                    
                    help2 = action_probs
                    
                    match state:
                        case 1: 
                            actual_V_new[state] = torch.dot(help2, self.reward_function[state, :]).item()
                        case 2:
                            help2 = action_probs[torch.nonzero(help2)].squeeze()
                            actual_V_new[state] = torch.dot(help2,torch.tensor([self.reward_function[state, 0] + gamma * actual_V_h[1], self.reward_function[state, 1] + gamma * actual_V_h[3], self.reward_function[state, 2] + gamma * actual_V_h[4]])).item()
                        case 4:
                            x2 = self.reward_function[state, :] + torch.ones_like(torch.tensor(self.reward_function[state, :]))* gamma * actual_V_h[5]
                            actual_V_new[state] = torch.dot(help2, x2).item()
                        case 5:
                            actual_V_new[state] = torch.dot(help2, self.reward_function[state, :]).item()
            
                actual_V_h = actual_V_new

                # save the value function for horizon (1...horizon-1) is never used when horizon = 1
                if h ==horizon-2:
                    optimal_V_h_old = actual_V_h.copy()

            # caluclate the best possible value function for the fixed future policies
            for i in range(4):
                    state = init_states[i]
                    match state:
                        case 1:
                            optimal_V_h[state] = torch.max(self.reward_function[state, :]).item()
                        case 2:
                            optimal_V_h[state] = torch.max(torch.tensor([self.reward_function[state, 0] + gamma * optimal_V_h_old[1], self.reward_function[state, 1] + gamma * optimal_V_h_old[3], self.reward_function[state, 2] + gamma * optimal_V_h_old[4]]).squeeze()).item()
                        case 4:
                            optimal_V_h[state] = torch.max(self.reward_function[state, :] + gamma * actual_V_h[5]).item()
                        case 5:
                            optimal_V_h[state] = torch.max(self.reward_function[state, :]).item()

            # averaging over columns (episodes)
            optimal_V_h= torch.tensor([optimal_V_h[s] for s in init_states]).squeeze()
            actual_V_h = torch.tensor([actual_V_h[s] for s in init_states]).squeeze()
            optimal_V_h_mean = torch.mean(optimal_V_h)
            actual_V_h_mean = torch.mean(actual_V_h)

            return optimal_V_h_mean, actual_V_h_mean
        
        else:
            raise ValueError("This function is not implemented for this environment")



