import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

TODAY = datetime.datetime.now().strftime("%d-%m-%Y")


def plot_rewards(data, out_dir, task_name, mdp_type):
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
            # The x_vals are the same across each run
            x_vals = data[alg]['0'][x_name]
            min_xmax = min(min_xmax, data[alg]['0'][x_name][-1])
            max_xmin = max(max_xmin, data[alg]['0'][x_name][0])
            # Combine rewards from each run element-wise
            rewards = list(zip(*(data[alg][i]["rewards"] for i in data[alg])))
            mean_rewards = np.array([np.mean(s) for s in rewards])
            # Standard error of the mean
            # std_err = np.array([np.std(s, ddof=1) / np.sqrt(len(s)) for s in rewards])
            # Standard deviation
            std = np.array([np.std(s) for s in rewards])

            plt.plot(x_vals, mean_rewards, "-o", markersize=2, label=alg, c=clrs[i])
            plt.fill_between(x_vals, mean_rewards-std, mean_rewards+std,
                alpha=0.3, facecolor=clrs[i])
            # plt.plot(data[alg][x_name], data[alg]["rewards"], "-o", markersize=2, label=alg)

        plt.legend(loc="lower right")
        plt.ylim(bottom=0., top=1.)
        # Truncate for consistent x-axis
        plt.xlim(left=max_xmin, right=min_xmax)
        # plt.xlim(left=0.)
        plt.xlabel(f"Number of {x_name}")
        plt.ylabel("Average Reward")

        plt.savefig(f"{out_dir}/{mdp_type}_{task_name}_rew_vs_{x_name}_{TODAY}.png")
        plt.clf()
