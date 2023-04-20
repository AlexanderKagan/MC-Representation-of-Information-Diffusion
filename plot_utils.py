import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import seaborn as sns


def plot_graph(graph, states, state_2_color={-1: "r", 1: "b", 0: "grey"}):
    colors = [state_2_color[state] for state in states]
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_color=colors, alpha=0.6, with_labels=True, arrows=True)


def plot_infection_times(infection_times, lambdas, mus, log_time=True):
    plt.figure(figsize=(9, 7))
    times_df = pd.DataFrame(columns=lambdas, index=mus, dtype=float)
    for time, (lmbda, mu) in zip(infection_times, product(lambdas, mus)):
        times_df.at[lmbda, mu] = time
    sns.heatmap(np.log(times_df) if log_time else times_df)
    plt.xlabel("$\lambda$", fontsize=12)
    plt.ylabel("$\mu$", fontsize=12)
    plt.title(("Log-" if log_time else "") + "Time vs ($\lambda, \mu$)")
    plt.show()


def plot_idea_full_spread_prob(idea_full_spread_prob, p_minus_plus_pairs):
    p_plus_range, p_minus_range = zip(*p_minus_plus_pairs)

    p_plus_range = sorted(list(set(p_plus_range)))
    p_minus_range = sorted(list(set(p_plus_range)))

    prob_df = pd.DataFrame(columns=p_plus_range, index=p_minus_range, dtype=float)
    for prob, (p_plus, p_minus) in zip(idea_full_spread_prob, p_minus_plus_pairs):
        prob_df.at[p_minus, p_plus] = prob

    plt.figure(figsize=(9, 7))

    sns.heatmap(prob_df)
    plt.xlabel("Proportion of idea spreaders", fontsize=12)
    plt.ylabel("Proportion of anti-idea spreaders", fontsize=12)
    plt.title("Idea full spread probability")
    plt.show()
