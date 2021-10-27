import matplotlib.pyplot as plt


def plot_rewards(data, out_dir, task_name):
    """
    Line plot of rewards vs number of (s,a,s') experiences or updates used. Since each algorithm
    will produce a varying number of experiences and updates, we truncate the plots to keep the
    scale of the x-range the same.
    """
    x_names = [x_name for alg in data for x_name in data[alg] if x_name != "rewards"]
    for x_name in x_names:
        plt.style.use("ggplot")
        # Get the smallest of the max xvals. Used for plot truncation
        min_xmax = 1e8
        for alg in data:
            min_xmax = min(min_xmax, data[alg][x_name][-1])
            plt.plot(data[alg][x_name], data[alg]["rewards"], "-o", markersize=6, label=alg)

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05))
        plt.ylim(bottom=0., top=1.)
        # Truncate for consisten x-axis
        plt.xlim(left=0., right=min_xmax)
        plt.xlabel(f"Number of {x_name}")
        plt.ylabel("Average Reward")

        plt.savefig(f"{out_dir}/{task_name}_{alg}_rew_vs_{x_name}.png")
        plt.clf()
