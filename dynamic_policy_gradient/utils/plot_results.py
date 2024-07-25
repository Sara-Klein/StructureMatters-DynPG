import numpy as np
import matplotlib.pyplot as plt
import os
import pickle



def plot_success_probs(max_interactions,discount_factor, rounds,pg_lr, dpg_lr, adaptive,env_name,eps, reward_list,Ns):
    discount_factor = discount_factor[0]
    plt.figure(figsize=(6, 5))
    total_samples_pg = []
    total_samples_dpg = []

    for i in range(rounds):
        #load prob for pg
        pickle_file_path = os.path.join("data", f"round={i}_alg=pg_env={env_name}_rewards={reward_list[0]}_eps={eps}_gamma={discount_factor}_pglr={pg_lr[0]}_dpr_lr={dpg_lr[0]}_adaptive={adaptive[0]}.pkl")
        with open(pickle_file_path, "rb") as f:
            data_dict = pickle.load(f)
        total_samples_pg.append(data_dict["total_samples"])
        #load prob for dpg
        pickle_file_path = os.path.join("data", f"round={i}_alg=dpg_env={env_name}_rewards={reward_list[0]}_eps={eps}_gamma={discount_factor}_pglr={pg_lr[0]}_dpr_lr={dpg_lr[0]}_adaptive={adaptive[0]}.pkl")
        with open(pickle_file_path, "rb") as f:
            data_dict = pickle.load(f)
        total_samples_dpg.append(data_dict["total_samples"])
    
    probs_pg = []
    probs_dpg = []
    for compute in range(max_interactions):
        probs_pg.append(np.sum([1 for i in total_samples_pg if i < compute]) / rounds)
        probs_dpg.append(np.sum([1 for i in total_samples_dpg if i < compute]) / rounds)


    # Plot 
    plt.plot(
        range(max_interactions),
        probs_pg,
        label="PG",
        color="orange",
        linewidth=2.5,
    )
    plt.plot(
        range(max_interactions),
        probs_dpg,
        label="DynPG",
        color="blue",
        linewidth=2.5,
    )

    # enlarge ticksize and label size
    plt.xlabel("Interactions with environment", fontsize=15)
    plt.ylabel("Success Probability", fontsize=15)
    plt.legend(loc="lower right", fontsize=15)

    # set ticksize
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1)

    plt.tight_layout()

    # Save the plot as an image file
    if adaptive[0]:
        plot_file_path = os.path.join("plots", f"sucess_plot_{env_name}_rewards={reward_list[0]}_eps={eps}_gamma={discount_factor}_rounds={rounds}_adaptive={adaptive[0]}.png")
    else:
        plot_file_path = os.path.join("plots", f"sucess_plot_{env_name}_rewards={reward_list[0]}_eps={eps}_gamma={discount_factor}_rounds={rounds}_pglr={pg_lr[0]}_dpglr={dpg_lr[0]}_dpgNs={Ns[0]}.png")

    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_file_path)

