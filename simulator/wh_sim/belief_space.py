import numpy as np
import random

from . import nnModel

class BeliefSpace:

    def __init__(self, mutation_rate=0.1, select_t=1, bank_size=5):
        self.store = []
        self.belief_bank = []
        self.mutation_rate = mutation_rate
        self.select_t=select_t
        self.bank_size = bank_size
        self.fitness_scores = []
        
        self.nn_model = None
        self.nn_xover_segment_l = None

    def init_nn_model2(self, input_size, output_size, hidden_nodes_per_layer, weights):
        self.nn_model = nnModel(input_size, output_size, hidden_nodes_per_layer, weights)
        self.store = weights
        self.nn_xover_segment_l = max(1, int(len(weights)/20))

    def init_nn_model(self,
                      input_size,
                      output_size,
                      hidden_nodes_per_layer,
                      weights=None):

        self.nn_model = nnModel(
            input_size,
            output_size,
            hidden_nodes_per_layer
        )

        # CASE 1: weights filename
        if isinstance(weights, str):
            self.nn_model.load_weights_txt(weights)
            self.store = self.nn_model.get_weights()

        # CASE 2: weights list
        elif isinstance(weights, list):
            self.nn_model.set_weights(weights)
            self.store = weights

        # CASE 3: no weights
        else:
            self.store = self.nn_model.get_weights()

        self.nn_xover_segment_l = max(1, int(len(self.store) / 20))

        
    def update_bank(self, new_belief, fitness):
        try:
            new_belief = new_belief.tolist()
        except:
            pass

        if len(self.belief_bank) == self.bank_size: # It's full
            self.belief_bank = self.belief_bank[1:]
            self.fitness_scores = self.fitness_scores[1:]

        self.belief_bank.append(new_belief)
        self.fitness_scores.append(fitness)

    def _crossover(self, x1, x2, segment_l=5):
        xover_0 = random.randrange(0,len(self.store))
        xover_1 = min(len(self.store),xover_0+segment_l)
        x1[xover_0:xover_1] = x2[xover_0:xover_1]
        return x1
    
    def _mutation(self, x):
        mask = np.random.binomial(1,self.mutation_rate,len(x))
        mutate_d = np.random.normal(0,0.1,1)
        return np.array(x) + np.array(x)*mask*mutate_d

    def update_from_bank(self):
        if len(self.belief_bank) == 0: # It didn't socialize yet - no new beliefs :)
            return
        
        # Select from belief bank
        selected = random.randrange(0,min(self.select_t,len(self.belief_bank)))
        selected_idx = np.argsort(self.fitness_scores)[::-1][selected]
        selected_belief = self.belief_bank[selected_idx]
        
        new_belief = self._crossover(self.store, selected_belief, self.nn_xover_segment_l)
        new_belief = self._mutation(new_belief)
        self.store = new_belief.tolist()
        self.nn_model.set_weights(new_belief)

    @staticmethod
    def normalize_input(x):

        return [
            x[0],
            x[1] / 1000.0,
            x[2] / 10.0,
            x[3] / 50.0,
            x[4]
        ]

    def generate_norm(self, input):
        output = np.array(self.nn_model.compute(self.normalize_input(input))) # Assume order of output is # [ag1_Pm1, ag1_Pm2, ag2_Pm1, ag2_Pm2,...]
        return {
            "P_m": output[::4].copy(),
            "D_m": output[1::4].copy(),
            "SC": output[2::4].copy(),
            "r0": output[3::4].copy()
        }
