from dynamic_policy_gradient.envs import FiniteMDP
import logging
import numpy as np

def sutton_example(discount_factor, seed =None, action_list = None, reward_list=None):
    # display the MDP information
    logging.info("------------Constructing The Sutton Example------------")
    logging.info(f"States: [0|1|2|3|4|5|6]; Actions: {action_list}; Rewards: {reward_list}")
    
    #fix seed
    if seed == None:
        seed = np.random.randint(0, 1000, (1,)).item() 
    
    # Instantiate the MDP environment
    env = FiniteMDP(
        env_id="Sutton-extended",
        n_states=7,
        n_actions=max(action_list),
        discount_factor=discount_factor,
        episode_length=500,
        varying_action_spaces=False,
        seed=seed,
    )

    # Set the goal state
    env.add_terminal_state(0)
    env.add_terminal_state(3)
    env.add_terminal_state(6)

    # Set all the transitions and rewards

    for a in range(action_list[1]):
        env.set_transition(1, a, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        env.set_reward(1, a, reward_list[1][0])
        env.set_reward_std(1, a, reward_list[1][1])

    env.set_transition(2, 0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.set_transition(2, 1, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env.set_transition(2, 2, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    env.set_reward(2,0, 0.0)
    env.set_reward(2,1, 0.0)
    env.set_reward(2,2, 0.0)
    
    for a in range(action_list[4]):
        env.set_transition(4, a, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        env.set_reward(4, a, reward_list[4][0])
        env.set_reward_std(4, a, reward_list[4][1])
    for a in range(action_list[5]):
        env.set_transition(5, a, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        env.set_reward(5, a, reward_list[5][0])
        env.set_reward_std(5, a, reward_list[5][1])
   

    logging.info("------------Construction Completed------------")
    return env


