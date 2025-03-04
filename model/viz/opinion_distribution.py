import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json

class OpinionDistribution(object):
    def __init__(self, 
                 trends: dict, 
                 p_o,
                 p_p,
                 iteration: int | str = -1, 
                 values: str = "probabilities"):
        
        """
        :param model: The model object
        :param trends: The computed simulation trends
        :param iteration: The iteration number or the string "last" for plotting final state
        """
        
        self.trends = trends
        self.iteration = iteration
        
        if iteration == 'last':
            self.it = self.trends[-1]['iteration']
            self.ops = self.trends[-1]['status']
        else:
            self.ops = self.trends[iteration]['status'].values()
        
        if values == 'probabilities':
            weights = np.array([el for el in self.ops])
            self.values = p_o * weights + p_p * (1-weights)
            
        elif values == 'weights':
            self.values = np.array([el for el in self.ops])
    
    def get_values(self):
        return self.values
