import numpy as np
import random

from . import nnModel

class BeliefSpace:

    def __init__(self, mutation_rate=0.1, select_t=1, bank_size=[]):
        self.store = []
        self.belief_bank = np.array([])
        self.mutation_rate = mutation_rate
        self.select_t=1
        self.bank_size = bank_size
        self.fitness_scores = np.array([])

        self.nn_model = None
        self.nn_xover_segment_l = None

    def init_nn_model(self, input_size, output_size, hidden_nodes_per_layer, weights):
        self.nn_model = nnModel(input_size, output_size, hidden_nodes_per_layer, weights)
        self.store = weights
        self.nn_xover_segment_l = max(1, int(len(weights)/10))
        
    def update_store(self, new_belief, fitness):
        self.belief_bank = self.belief_bank[1:]
        self.belief_bank.append(new_belief)
        self.fitness_scores = self.fitness_scores[1:]
        self.fitness_scores.append(fitness)

    def _crossover(self, x1, x2, segment_l=5):
        xover_0 = random.randrange(0,len(self.store))
        xover_1 = min(len(self.store),xover_0+segment_l)
        x1[xover_0:xover_1] = x2[xover_0:xover_1]
        return x1
    
    def _mutation(self, x):
        mask = np.random.binomial(1,self.mutation_rate,len(x))
        mutate_d = np.random.normal(0,0.1,1)
        if mutate_d > 0:
            mutate_d += 1
        else:
            mutate_d -= 1
        return np.array(x)*mask*mutate_d

    def update_from_crossover(self):
        # Select from belief bank
        selected = random.randrange(0,self.select_t)
        selected_idx = np.argsort(self.fitness_scores)[::-1][selected]
        selected_belief = self.belief_bank[selected_idx]

        new_belief = self._crossover(self.store, selected_belief, self.nn_xover_segment_l)
        new_belief = self._mutation(new_belief)
        self.store = new_belief

    def generate_norm(self, input):
        output = self.nn_model.compute(input)
        # translate output to param space
        return output
