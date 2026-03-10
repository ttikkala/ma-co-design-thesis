import numpy as np

"""
This script contains a class implementing the Modified Particle Swarm Optimisation method of 
Berthold Immanuel Schmitt "Convergence Analysis for Particle Swarm Optimization."
PhD thesis, Friedrich-Alexander-Universit¨at Erlangen-N ¨urnberg (FAU) Technische Fakultaet, 2015.

With the boundary handling method "Reflect-Z" from S. Helwig et al. "Experimental Analysis of Bound Handling
Techniques in Particle Swarm Optimization." IEEE Transactions on Evolutionary Computation, 17(2):259–271, 2013.
"""

class PSOModified():

    def __init__(self, n_particles, dimensions, bounds, options):
        """
        """
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.lower = bounds[0][0]
        self.upper = bounds[1][0]

        self.c1 = options['c1']
        self.c2 = options['c2']
        self.w = options['w']
        self.positions = np.zeros(shape=(self.n_particles, self.dimensions))
        self.velocities = np.zeros(shape=(self.n_particles, self.dimensions))

        for i in range(n_particles):
            self.positions[i] = np.random.uniform(low=self.lower, high=self.upper, size=self.dimensions)
            self.velocities[i] = np.random.uniform(low=-(self.upper - self.lower), high=(self.upper - self.lower),
                                                   size=self.dimensions)
            
        self.delta = 1e-6

        np.set_printoptions(threshold=100000)


    def optimize(self, f, print_step, iters, verbose=None):
        """
        """

        idx = 0
        global_best_cost = np.inf
        local_best_costs = np.inf * np.ones(self.n_particles)
        global_best_pos = np.random.uniform(low=self.lower, high=self.upper, size=self.dimensions)
        local_best_positions = self.positions

        while idx < iters:
            for i in range(self.n_particles):
                for j in range(self.dimensions):
                    rp = np.random.uniform()
                    rg = np.random.uniform()

                    if ((np.abs(self.velocities[:,j]) + np.abs(global_best_pos[j] - self.positions[:,j])) < self.delta).all():
                        self.velocities[:,j] = (2 * rp - 1) * self.delta
                    else:
                        self.velocities[i][j] = (self.w * self.velocities[i][j] 
                                                 + self.c1 * rp * (local_best_positions[i][j] - self.positions[i][j]) 
                                                 + self.c2 * rg * (global_best_pos[j] - self.positions[i][j]))
                
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # Boundary handling - reflect-Z
                lower_violations = self.positions[i] < self.lower
                upper_violations = self.positions[i] > self.upper

                self.positions[i][lower_violations] = self.lower + (self.lower - self.positions[i][lower_violations])
                self.positions[i][upper_violations] = self.upper - (self.positions[i][upper_violations] - self.upper)

                self.velocities[i][lower_violations] = 0
                self.velocities[i][upper_violations] = 0

                lower_violations = self.positions[i] < self.lower
                upper_violations = self.positions[i] > self.upper

                while lower_violations.any() or upper_violations.any():
                    self.positions[i][lower_violations] = self.lower + (self.lower - self.positions[i][lower_violations])
                    self.positions[i][upper_violations] = self.upper - (self.positions[i][upper_violations] - self.upper)
                    lower_violations = self.positions[i] < self.lower
                    upper_violations = self.positions[i] > self.upper


            # Calculate cost
            costs = f(self.positions)

            for i in range(self.n_particles):
                # Update global best
                if costs[i] < local_best_costs[i]:
                    local_best_costs[i] = costs[i]
                    local_best_positions[i] = self.positions[i].copy()

                    if costs[i] < global_best_cost:
                        global_best_cost = costs[i]
                        global_best_pos = self.positions[i].copy()


            if idx % print_step == 0 or idx == 0:
                print("Iteration: ", idx)
                print("Best cost: ", global_best_cost)
                print("Best position: ", global_best_pos)
                print("All costs: ", costs)
            
            idx += 1


        return global_best_cost, global_best_pos
    
