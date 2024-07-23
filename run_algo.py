from dynamic_policy_gradient.utils import (
    sutton_example,
    setup_logging,
    change_log_file
)

from absl import app
from dynamic_policy_gradient.utils.plot_results import plot_success_probs
from dynamic_policy_gradient.algs import SoftmaxPG, DynamicSoftmaxPG
import itertools
import configs_getter
import torch
import os
import pickle


_ALGOS = {
    "pg": SoftmaxPG,
    "dpg" : DynamicSoftmaxPG
}

_ENVS = {
    "Sutton-extended": sutton_example
}

logging =True
if logging:
    setup_logging()
    



def _run_algos():
    for config_name, config in configs_getter.get_configs():
        print(f'Config {config_name}', config)
        combinations = list(itertools.product(
            config.algorithm, config.env_name, config.action_list, config.reward_list,config.discount_factor,  
            config.eps, config.pg_lr, config.dpg_lr, config.adaptive, config.dpg_Ns, 
            config.computational_power, config.gradient_clipping, config.reward_normalization, 
            config.copy_weights, config.rounds
        ))
        
        for params in combinations:
            _run_algo(*params)

    
    plot_success_probs(config.computational_power[0], config.discount_factor, 
                        config.rounds[0], config.pg_lr, config.dpg_lr,config.adaptive,
                        config.env_name[0], config.eps[0], config.reward_list)
            

def _run_algo(algorithm, env_name, action_list, reward_list, discount_factor, 
              eps, pg_lr, dpg_lr, adaptive, dpg_Ns, computational_power,
              gradient_clipping, reward_normalization, copy_weights,
              rounds):
    # one can speficify seeds for reproducibility; 
    # if seed is None, the seed is randomly generated
    seed = None
    
    
    for i in range(rounds):
        if logging:
            lr_to_record = pg_lr if algorithm == "pg" else dpg_lr
            change_log_file(
                os.path.join(
                    "logs",
                    f"{env_name}_{algorithm}_eps={eps}_gamma={discount_factor}_lr={lr_to_record}.log",
                )
            )
        env = _ENVS[env_name](discount_factor,seed, action_list, reward_list)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
        alg = _ALGOS[algorithm](env, device, action_list)

        match algorithm:
            case "pg": 
                alg.train(
                eps,
                pg_lr,
                discount_factor,
                gradient_clipping,
                reward_normalization,
                computational_power,
                adaptive
                )
            case "dpg":
                alg.train(
                eps,
                dpg_lr,
                adaptive,
                dpg_Ns,
                discount_factor,
                gradient_clipping,
                reward_normalization,
                copy_weights,
                computational_power,
                )
            case _:
                raise ValueError(f"Algorithm {alg} not found")
    
    

        os.makedirs("data", exist_ok=True)
    
        # dump the converged policy into pickle files
        data_dict = {
            "episode_rewards": alg.episode_rewards,
            "episode_V_gaps": alg.episode_V_gaps,
            "episode_actual_Vs": alg.episode_actual_Vs,
            "total_episodes": alg.total_episodes,
            "total_samples": alg.total_samples,
            "avg_samples_per_episode": alg.avg_samples_per_episode,
            "sample_path": alg.sample_path,
        }

        pickle_file_path = os.path.join(
            "data", f"round={i}_alg={algorithm}_env={env_name}_rewards={reward_list}_eps={eps}_gamma={discount_factor}_pglr={pg_lr}_dpr_lr={dpg_lr}_adaptive={adaptive}.pkl"
        )
        with open(pickle_file_path, "wb") as f:
            pickle.dump(data_dict, f)




def main(argv):
    del argv
    filepath =_run_algos()

if __name__ == "__main__":
    app.run(main)