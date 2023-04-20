import numpy as np
import networkx as nx
from numpy.random import rand, choice


class MarkovInfectionModel:
    def __init__(self, graph: nx.DiGraph, lmbda=0.5, mu=0.5):
        self.nodes = np.array(list(graph.nodes))
        assert np.all(np.sort(self.nodes) == np.arange(len(self.nodes))), "nodes should be named from 0 to |V| - 1"
        self.graph = graph
        self.lmbda = lmbda
        self.mu = mu
        self.states = None
        self.infection_time = None

    def _infect_neighbors(self, seed: int) -> None:
        """
        :param seed: int, vertex spreading infection
        """
        infection = self.states[seed]
        assert infection != 0, "seed is indifferent"

        neighbors = np.array(list(self.graph.neighbors(seed)))
        if neighbors.size == 0:
            return

        indifferent_neighbors_mask = self.states[neighbors] == 0
        if np.any(indifferent_neighbors_mask):
            indifferent_neighbors = neighbors[indifferent_neighbors_mask]
            indifferent_res = np.where(rand(len(indifferent_neighbors)) <= self.lmbda, infection, 0)
            self.states[indifferent_neighbors] = indifferent_res

        different_neighbors_mask = self.states[neighbors] == -infection
        if np.any(different_neighbors_mask):
            different_neighbors = neighbors[different_neighbors_mask]
            different_res = np.where(rand(len(different_neighbors)) <= self.mu, infection, -infection)
            self.states[different_neighbors] = different_res

    def make_simulation(self, n_messages, init_states, verbose=False, verbose_interval=1):
        self.states = np.array(init_states, dtype=int)
        for message in range(n_messages):
            seed = choice(self.nodes[self.states != 0])
            self._infect_neighbors(seed)
            if verbose and message % verbose_interval == 0:
                print(f"Message: {message}, Seed: {seed}, State: {list(self.states)}")
            if np.all(self.states == self.states[0]):
                if verbose:
                    print(f"All nodes become infected after {message} messages")
                self.infection_time = message
                break

        return self.states

    def reward(self):
        return np.sum(self.states == 1)
