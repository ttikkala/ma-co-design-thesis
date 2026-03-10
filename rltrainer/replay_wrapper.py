import random

"""
This script contains a helper class that samples the population replay buffer roughly 30% of the time
and the individual replay buffer 70% of the time.
"""

class ReplayWrapper:

    def __init__(self, ind_replay, pop_replay):
        self.ind_replay = ind_replay
        self.pop_replay = pop_replay

    def random_batch(self, batch_size):
        if random.randint(0,9) < 3:
            return self.pop_replay.random_batch(batch_size)
        else:
            return self.ind_replay.random_batch(batch_size)