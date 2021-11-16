import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_rewards(data, out_dir, task_name, mdp_type, optimal_reward, date):
    """
    Line plot of rewards vs number of (s,a,s') experiences or updates used. Since each algorithm
    will produce a varying number of experiences and updates, we truncate the plots to keep the
    scale of the x-range the same.
    """
    clrs = sns.color_palette("husl", 8)
    for x_name in ["samples", "updates"]:
        plt.style.use("ggplot")
        # Used for plot truncation
        # Get the smallest of the max xvals.
        min_xmax = 1e8
        # Get the largest of the min xvals
        max_xmin = 0
        for i, alg in enumerate(data):
            x_vals = data[alg][x_name]
            min_xmax = min(min_xmax, data[alg][x_name][-1])
            max_xmin = max(max_xmin, data[alg][x_name][0])
            # Combine rewards from each run element-wise
            rewards = list(zip(*data[alg]["rewards"]))
            quartiles = np.array([np.percentile(s, [25, 50, 75]) for s in rewards])

            # Plot the performance of the median run
            plt.plot(x_vals, quartiles[:,1], "-o", markersize=2, label=alg, c=clrs[i])
            # Plot a shaded region for the 25% and 75% quartiles
            plt.fill_between(x_vals, quartiles[:,0], quartiles[:,2], alpha=0.2, facecolor=clrs[i])

        plt.ylim(bottom=0., top=1.)
        plt.axhline(y=optimal_reward, color='k', linestyle='-', label="optimal av. reward")
        # plt.yticks(list(plt.yticks()[0]) + [optimal_reward])
        # Truncate for consistent x-axis
        plt.xlim(left=0, right=min_xmax)
        plt.xlabel(f"Number of {x_name}")
        plt.ylabel("Average Reward")
        plt.legend(loc="lower right")

        plt.savefig(f"{out_dir}/{mdp_type}_{task_name}_rew_vs_{x_name}_{date}.png")
        plt.clf()