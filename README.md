# StructureMatters-DynPG
 
 Implementation of DynPG for discounted MDPs to test the performance compared to vanilla PG.

 ## How to get started

Use "pip install -e ." to init the folder.

Execute the file "run_algo.py" to start the experiment. You can find the generated plot in the folder plots.

# How to run different configurations

You can modify the config.py file 

- In action_list you can specify the number of arms. Modification of entry 1, 4 or 5 is allowed
- By changing the non-zero entries in reward_list one can try different means and standard-deviations of the normally distributed arms
- discount_factor: specify $\gamma$ of the MDP
- eps: set the accuracy $\epsilon$ at which training is completed (overall error in the DynPG Paper less than $\epsilon$)
- You can manually choose the step size for PG or DynPG as well as the number of training steps for DynPG (then adaptive has to be False)
- computational_power: Maximal interactions with the environment (Algo stops when total samples > computational_power)
- rounds: Number of runs to estimate the success probabilities

