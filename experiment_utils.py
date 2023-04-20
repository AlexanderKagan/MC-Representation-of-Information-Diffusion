from InfectionModel import MarkovInfectionModel
import numpy as np


def make_init_state(n_nodes, p_plus: float, p_minus: float):
    """
    p_plus: proportion of infected with idea
    p_minus: proportion of infected with anti-idea
    """
    assert p_plus + p_minus <= 1, "sum of proportions should not exceed 1"
    n_plus = int(n_nodes * p_plus)
    n_minus = int(n_nodes * p_minus)
    n_indif = n_nodes - n_plus - n_minus
    init_state = np.hstack([[-1] * n_minus, [1] * n_plus, [0] * n_indif])
    return init_state


def run_time_experiment(rate_params, g, init_state, n_simulations=10000, n_messages=1000):
    lmbda, mu = rate_params
    infection_times = []

    for _ in range(n_simulations):
        model = MarkovInfectionModel(g, lmbda=lmbda, mu=mu)
        model.make_simulation(n_messages, init_state, verbose=False)
        if model.infection_time is not None:
            infection_times.append(model.infection_time)

    #     print("Proportion of fully infected processes", len(infection_times) / n_simulations)
    return np.mean(infection_times)


def run_full_spread_experiment(proportion_params, model: MarkovInfectionModel,
                               n_simulations: int = 10000, n_messages: int = 1000):
    p_plus, p_minus = proportion_params
    idea_full_spread = 0
    anti_idea_full_spread = 0

    for _ in range(n_simulations):
        init_state = make_init_state(model.graph.number_of_nodes(), p_plus, p_minus)
        final_states = model.make_simulation(n_messages, init_state)

        if np.all(final_states == 1):
            idea_full_spread += 1
        elif np.all(final_states == -1):
            anti_idea_full_spread += 1

    return idea_full_spread / (idea_full_spread + anti_idea_full_spread)
