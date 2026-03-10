import numpy as np
import torch

""" 
    This script contains the observation normaliser class which handles normalising new observations
    based on a simple running average updated throughout training.
"""

class ObservationNormaliser:
    def __init__(self, shape, epsilon=1e-16, device='cuda'):
        self.shape = shape
        self.means = torch.zeros(self.shape, device=device)
        self.vars = torch.ones(self.shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon
        self.M2s = torch.zeros(self.shape, device=device)
        self.sum = torch.zeros(self.shape, device=device)
        self.sum_sq = torch.zeros(self.shape, device=device)
    
    def reset(self):
        self.means = torch.zeros(self.shape, device='cuda')
        self.vars = torch.ones(self.shape, device='cuda')
        self.count = self.epsilon
        self.epsilon = self.epsilon
        self.M2s = torch.zeros(self.shape, device='cuda')
        self.sum = torch.zeros(self.shape, device='cuda')
        self.sum_sq = torch.zeros(self.shape, device='cuda')
    

    # def update_one(self, obs):
    #     self.count += 1
    #     deltas = obs - self.means
    #     self.means += deltas / self.count
    #     delta2s = obs - self.means
    #     self.M2s += deltas * delta2s
    #     self.vars = self.M2s / self.count
    
    def update_batch(self, batch):
        self.sum += torch.sum(batch, dim=0)
        self.sum_sq += torch.sum(batch**2, dim=0)
        self.count += batch.size(0)

        self.means = self.sum / self.count
        self.vars = (self.sum_sq / self.count) - self.means**2
        self.vars = torch.clamp(self.vars, min=self.epsilon)


    def get_means_vars(self):
        if self.count < 2:
            return float("nan")
        else:
            (means, variances) = (self.means, self.vars)
            return (means, variances)
    
    def normalise(self, obs):
        return (obs - self.means) / torch.sqrt(self.vars + self.epsilon)
