import matplotlib.pyplot as plt


def plot_rewards(data, x_name, out_dir):
    """Line plot of rewards vs number of (s,a,s') experiences or updates used."""
    # Create plot and save to directory
    plt.style.use("ggplot")
    for alg_name in data:
        plt.plot(data[alg_name][x_name], data[alg_name]["rewards"], label=alg_name)
    # plt.plot(data["num_examples"], data["cv_losses"], "-o", label="Validation Loss")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.ylim(top=1.)
    plt.xlabel(f"Number of {x_name}")
    plt.ylabel("Average Reward")

    plt.savefig(f"{out_dir}/{alg_name}_rew_vs_{x_name}.png")
    plt.clf()
