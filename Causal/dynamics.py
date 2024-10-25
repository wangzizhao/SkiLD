import copy
from torch import nn
import numpy as np

class Dynamics(nn.Module):
    '''
    Returns the ground truth causal graph by accessing data.true_graph
    '''
    def __init__(self, env, extractor):
        super().__init__()
        # initialize necessary components
        pass
    
    def __call__(self, data):
        '''
        returns the graph, as computed with the necessary information in data
        '''
        return 

    def update(self, batch_size, buffer):
        '''
        trains the dynamisc function, using batch_size @param batches
        sampled from the @param buffer, on @param repeat number of iterations 
        returns loss dictionary
        '''
        return dict()

    def compute_weight(self, data, dynamics, graph, true_graph, proximity):
        '''
        returns an unnormalized weight value to assign to the current value
        used for training the dynamics network
        this might be computed using proximity, the graph, or the true graph
        defaults to equal weight for every data point
        '''
        return np.ones((len(data), ))
